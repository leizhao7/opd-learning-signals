# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import contextlib
import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        # Allow a controlled SDPA-kernel isolation without changing the model
        # implementation.  PyTorch's default SDPA dispatcher may select its
        # fused flash or cuDNN kernel; forcing the efficient or math backend
        # lets numerical smoke tests distinguish those kernels precisely.
        sdpa_backend = os.getenv("ACTOR_SDPA_BACKEND", "default").lower()
        if sdpa_backend in {"efficient", "math"}:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(sdpa_backend == "efficient")
            torch.backends.cuda.enable_math_sdp(sdpa_backend == "math")
        elif sdpa_backend != "default":
            raise ValueError(
                f"Unsupported ACTOR_SDPA_BACKEND={sdpa_backend!r}; "
                "expected default, efficient, or math."
            )
        if torch.distributed.get_rank() == 0:
            print(f"{role} sdpa_backend={sdpa_backend}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, top_k=0, student_top_k_ids=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
        """
        response_length = micro_batch["responses"].size(-1)
        # Sampling temperature 0 means greedy decoding. Policy log-probabilities
        # are still defined by the untempered model distribution; dividing logits
        # by zero here would create NaNs in forward-OPD rewards and gradients.
        logprob_temperature = temperature if temperature is not None and temperature > 0 else 1.0
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = logprob_temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(logprob_temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        # Gather log_probs for target tokens
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)
                                 
                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = logprob_temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(logprob_temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        # Gather log_probs for target tokens (responses)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

            return entropy, log_probs, topk_ids, topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")
        # Modes:
        #   student_p / teacher_p / none: legacy log-ratio reward weighting.
        #   forward_kl: direct teacher cross-entropy gradient on the selected
        #               support.  With token_reward_direct and ppo_epochs=1,
        #               positive teacher-probability advantages yield
        #               -sum_a q(a) grad log p_theta(a), i.e. forward KL.
        is_forward_kl = reward_weight_mode == "forward_kl"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", "forward_kl", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode in ("teacher_p", "forward_kl"):
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        def form_distillation_scores(kl_val, weights):
            if is_forward_kl:
                # The actor minimizes -advantage * log p.  Returning q-weights
                # therefore implements teacher cross entropy directly; keeping
                # the log-ratio factor here would not be a forward-KL gradient.
                return weights
            return -kl_val * weights

        res_tensors = {}
        # Keep the exact support used by the forward objective so that we can
        # report its numerical value in addition to the PPO surrogate.  The
        # latter has the same gradient on policy, but its scalar value is not a
        # KL (and is typically negative when the advantages are positive).
        diagnostic_student_logp = None
        diagnostic_teacher_logp = None
        diagnostic_valid_mask = None
        diagnostic_teacher_weights = None
        
        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = form_distillation_scores(kl_val, norm_weights)
            diagnostic_student_logp = S_logp
            diagnostic_teacher_logp = T_on_S
            diagnostic_valid_mask = valid_mask
            diagnostic_teacher_weights = norm_weights
            
        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = form_distillation_scores(kl_val, norm_weights)
            diagnostic_student_logp = S_on_T
            diagnostic_teacher_logp = T_logp
            diagnostic_valid_mask = valid_mask
            diagnostic_teacher_weights = norm_weights
            res_tensors["union_top_k_ids"] = T_ids
            
        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = form_distillation_scores(kl_val, norm_weights)
            diagnostic_student_logp = S_logp
            diagnostic_teacher_logp = T_on_S
            diagnostic_valid_mask = valid_mask
            diagnostic_teacher_weights = norm_weights
            
        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = form_distillation_scores(kl_val, norm_weights)
            diagnostic_student_logp = S_logp_union
            diagnostic_teacher_logp = T_logp_union
            diagnostic_valid_mask = valid_mask
            diagnostic_teacher_weights = norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T
        
        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)
                
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = form_distillation_scores(kl_val, norm_weights)
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T

        if (
            is_forward_kl
            and diagnostic_student_logp is not None
            and diagnostic_teacher_logp is not None
            and diagnostic_valid_mask is not None
            and diagnostic_teacher_weights is not None
        ):
            # q is the teacher distribution renormalized on the selected
            # support.  p is the student's full-vocabulary distribution,
            # evaluated on that same support.  Extending q by zeros outside the
            # support makes this a genuine non-negative KL(q || p):
            #   KL = sum_a q(a) [log q(a) - log p(a)] = CE - H(q).
            # Compute diagnostics in float64 to keep CE-H non-negative up to
            # numerical roundoff, then return compact float tensors.
            valid = diagnostic_valid_mask.bool()
            q = diagnostic_teacher_weights.double()
            student_logp = torch.where(
                valid,
                diagnostic_student_logp.double(),
                torch.zeros_like(diagnostic_student_logp, dtype=torch.float64),
            )
            teacher_logp = torch.where(
                valid,
                diagnostic_teacher_logp.double(),
                torch.full_like(diagnostic_teacher_logp, -float("inf"), dtype=torch.float64),
            )
            log_q = torch.where(q > 0, torch.log(q), torch.zeros_like(q))
            forward_ce_token = -(q * student_logp).sum(dim=-1)
            teacher_entropy_token = -(q * log_q).sum(dim=-1)
            forward_kl_raw_token = forward_ce_token - teacher_entropy_token
            # Only sub-ulp negatives are possible for valid distributions.
            forward_kl_token = forward_kl_raw_token.clamp_min(0.0)
            student_support_mass_token = (
                torch.exp(student_logp) * valid.to(dtype=torch.float64)
            ).sum(dim=-1)
            teacher_support_mass_token = torch.exp(teacher_logp).sum(dim=-1)

            # Also expose the exact normalization gap that makes the legacy
            # top-k reverse PG scalar differ from a conditional reverse KL.
            # If P and Q are the student/teacher masses on the support S, then
            #   reverse_surrogate
            #     = KL(p(.|S) || q(.|S)) + log(P / Q).
            # This does not estimate the omitted full-vocabulary tail term; it
            # isolates the part of the discrepancy observable from top-k data.
            masked_student_logp = torch.where(
                valid,
                diagnostic_student_logp.double(),
                torch.full_like(diagnostic_student_logp, -float("inf"), dtype=torch.float64),
            )
            student_log_mass = torch.logsumexp(masked_student_logp, dim=-1)
            teacher_log_mass = torch.logsumexp(teacher_logp, dim=-1)
            has_support = valid.any(dim=-1)
            student_log_p_cond = masked_student_logp - student_log_mass.unsqueeze(-1)
            student_p_cond = torch.where(
                valid & has_support.unsqueeze(-1),
                torch.exp(student_log_p_cond),
                torch.zeros_like(student_log_p_cond),
            )
            reverse_log_ratio = torch.where(
                valid, student_logp - teacher_logp, torch.zeros_like(student_logp)
            )
            reverse_conditional_log_ratio = torch.where(
                valid,
                student_log_p_cond - log_q,
                torch.zeros_like(student_log_p_cond),
            )
            reverse_surrogate_token = (student_p_cond * reverse_log_ratio).sum(dim=-1)
            reverse_conditional_kl_token = (
                student_p_cond * reverse_conditional_log_ratio
            ).sum(dim=-1)
            reverse_log_mass_correction_token = student_log_mass - teacher_log_mass
            reverse_surrogate_token = torch.where(
                has_support, reverse_surrogate_token, torch.zeros_like(reverse_surrogate_token)
            )
            reverse_conditional_kl_token = torch.where(
                has_support,
                reverse_conditional_kl_token.clamp_min(0.0),
                torch.zeros_like(reverse_conditional_kl_token),
            )
            reverse_log_mass_correction_token = torch.where(
                has_support,
                reverse_log_mass_correction_token,
                torch.zeros_like(reverse_log_mass_correction_token),
            )

            res_tensors["forward_kl_token"] = forward_kl_token.float()
            res_tensors["forward_kl_raw_token"] = forward_kl_raw_token.float()
            res_tensors["forward_ce_token"] = forward_ce_token.float()
            res_tensors["forward_teacher_entropy_token"] = teacher_entropy_token.float()
            res_tensors["forward_student_support_mass_token"] = student_support_mass_token.float()
            res_tensors["forward_teacher_support_mass_token"] = teacher_support_mass_token.float()
            res_tensors["reverse_topk_surrogate_token"] = reverse_surrogate_token.float()
            res_tensors["reverse_conditional_kl_token"] = reverse_conditional_kl_token.float()
            res_tensors["reverse_log_mass_correction_token"] = reverse_log_mass_correction_token.float()
        
        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor c_par", logger=logger)
    def _measure_c_par(self, data: DataProto, temperature: float) -> dict[str, float]:
        """Measure cross-fit reward/OPD gradient alignment before any update.

        For an orientation ``a -> b``, fold ``a`` supplies a sampled-token
        REINFORCE loss and fold ``b`` supplies the token-population OPD policy
        objective implemented by this actor.  Both are minimization losses, hence their
        gradient dot product equals reward-ascent dot OPD-descent (the two minus
        signs cancel).  FSDP FULL_SHARD leaves a disjoint, already-reduced
        gradient shard on each rank; the reward shard is kept on CPU while the
        fresh OPD backward runs, and only three scalar sufficient statistics
        are all-reduced.
        """
        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("actor.c_par requires an initialized distributed process group.")
        if not isinstance(self.actor_module, FSDP):
            raise NotImplementedError("actor.c_par currently supports FSDP1 FULL_SHARD only.")
        if self.actor_module.sharding_strategy != ShardingStrategy.FULL_SHARD:
            raise NotImplementedError("actor.c_par requires fsdp_size=-1 / FULL_SHARD.")
        if int(self.ulysses_sequence_parallel_size) != 1:
            raise NotImplementedError("actor.c_par currently requires ulysses_sequence_parallel_size=1.")
        if int(self.config.ppo_epochs) != 1:
            raise NotImplementedError("actor.c_par currently requires ppo_epochs=1.")
        if int(self.config.rollout_n) != 1:
            raise NotImplementedError("actor.c_par is the single-rollout estimator and requires rollout.n=1 (G=1).")
        local_batch_size = int(data.batch.batch_size[0])
        if local_batch_size > int(self.config.ppo_mini_batch_size):
            raise NotImplementedError(
                "actor.c_par requires one on-policy mini-batch per update; "
                f"local batch={local_batch_size}, ppo_mini_batch_size={self.config.ppo_mini_batch_size}."
            )
        if str(self.config.loss_agg_mode) != "token-mean":
            raise NotImplementedError("actor.c_par currently requires loss_agg_mode=token-mean.")
        if float(self.config.entropy_coeff) != 0.0 or bool(self.config.use_kl_loss):
            raise NotImplementedError("actor.c_par measures pure practical OPD; entropy and KL-loss must be disabled.")
        if self.config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
            raise NotImplementedError("actor.c_par currently supports policy_loss.loss_mode=vanilla only.")
        if "multi_modal_inputs" in data.non_tensor_batch:
            raise NotImplementedError("actor.c_par does not yet support multimodal actor batches.")

        required_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "advantages",
            "c_par_fold",
            "c_par_reward_advantages",
        ]
        optional_keys = [
            "student_top_k_ids",
            "union_top_k_ids",
            "format_mask",
            "rollout_is_weights",
        ]
        missing = [key for key in required_keys if key not in data.batch]
        if missing:
            raise KeyError(f"actor.c_par input is missing tensor keys: {missing}.")
        selected_keys = required_keys + [key for key in optional_keys if key in data.batch]
        selected = data.select(batch_keys=selected_keys)

        folds = selected.batch["c_par_fold"].long()
        if not bool(torch.all((folds == 0) | (folds == 1)).item()):
            raise ValueError("c_par_fold must contain only 0 and 1.")
        response_mask = selected.batch["response_mask"].float()
        effective_mask = response_mask
        if "format_mask" in selected.batch:
            effective_mask = effective_mask * selected.batch["format_mask"].float().unsqueeze(-1)

        device = torch.device(self.device_name, get_device_id())
        world_size = dist.get_world_size()
        counts = torch.tensor(
            [
                float((folds == 0).sum().item()),
                float((folds == 1).sum().item()),
                float((effective_mask * (folds == 0).unsqueeze(-1)).sum().item()),
                float((effective_mask * (folds == 1).unsqueeze(-1)).sum().item()),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if bool(torch.any(counts <= 0).item()):
            raise ValueError(f"actor.c_par has an empty global fold or token set: {counts.tolist()}.")
        global_sequences = [float(counts[0].item()), float(counts[1].item())]
        global_tokens = [float(counts[2].item()), float(counts[3].item())]

        def make_micro_batches():
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(selected, max_token_len=max_token_len)
                return micro_batches
            return selected.split(int(self.config.ppo_micro_batch_size_per_gpu))

        parameters = [parameter for parameter in self.actor_module.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("actor.c_par found no trainable actor parameters.")

        requested_fold = int(data.meta_info.get("c_par_reward_fold", 0))
        symmetric = bool(data.meta_info.get("c_par_symmetric", False))
        reward_folds = (0, 1) if symmetric else (requested_fold,)
        if any(fold not in (0, 1) for fold in reward_folds):
            raise ValueError(f"Invalid c_par_reward_fold={requested_fold}.")

        directional: dict[int, tuple[float, float, float, float]] = {}
        self.actor_optimizer.zero_grad(set_to_none=True)
        try:
            for reward_fold in reward_folds:
                opd_fold = 1 - reward_fold

                # Fresh sampled-logprob REINFORCE gradient.  FSDP averages rank
                # gradients, so world_size/global_count produces the global mean.
                reward_scale = float(world_size) / global_sequences[reward_fold]
                for micro_batch in make_micro_batches():
                    micro_batch = micro_batch.to(device)
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    micro_mask = model_inputs["response_mask"].float()
                    fold_weight = (model_inputs["c_par_fold"].long() == reward_fold).float()
                    _, sampled_log_prob, _, _ = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                    )
                    sequence_log_prob = (sampled_log_prob.float() * micro_mask).sum(-1)
                    reward_loss = -(
                        model_inputs["c_par_reward_advantages"].float()
                        * sequence_log_prob
                        * fold_weight
                    ).sum() * reward_scale
                    reward_loss.backward()

                reward_gradient_shards = [
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().to(device="cpu", dtype=torch.float32, copy=True)
                    for parameter in parameters
                ]
                self.actor_optimizer.zero_grad(set_to_none=True)

                # Fresh token-mean OPD backward on the independent fold. Passing
                # current log-probs as detached old log-probs gives the on-policy
                # ppo_epochs=1 objective. This deliberately removes incidental
                # dynamic-microbatch weighting; it is a raw empirical-objective
                # gradient, not an Adam-preconditioned parameter displacement.
                opd_scale = float(world_size) / global_tokens[opd_fold]
                for micro_batch in make_micro_batches():
                    micro_batch = micro_batch.to(device)
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    advantages = model_inputs["advantages"]
                    base_mask = model_inputs["response_mask"].float()
                    fold_weight = (model_inputs["c_par_fold"].long() == opd_fold).float().unsqueeze(-1)
                    loss_mask = base_mask * fold_weight
                    format_mask = model_inputs.get("format_mask", None)

                    if advantages.dim() == 3:
                        top_k = int(advantages.shape[-1])
                        target_ids = model_inputs.get("union_top_k_ids", model_inputs.get("student_top_k_ids", None))
                        if target_ids is None:
                            raise KeyError("3D OPD advantages require student_top_k_ids or union_top_k_ids.")
                        _, _, _, log_prob_for_loss = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=False,
                            top_k=top_k,
                            student_top_k_ids=target_ids,
                        )
                    else:
                        _, log_prob_for_loss, _, _ = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=False,
                        )

                    policy_loss_fn = get_policy_loss_fn("vanilla")
                    pg_loss, _ = policy_loss_fn(
                        old_log_prob=log_prob_for_loss.detach(),
                        log_prob=log_prob_for_loss,
                        advantages=advantages,
                        response_mask=loss_mask,
                        loss_agg_mode="token-mean",
                        config=self.config,
                        rollout_is_weights=model_inputs.get("rollout_is_weights", None),
                        format_mask=format_mask,
                    )
                    micro_effective_mask = loss_mask
                    if format_mask is not None:
                        micro_effective_mask = micro_effective_mask * format_mask.float().unsqueeze(-1)
                    micro_tokens = micro_effective_mask.sum().detach()
                    (pg_loss * micro_tokens * opd_scale).backward()

                # Sum dot/norm sufficient statistics over disjoint FULL_SHARD
                # shards.  A parameter-sized reward tensor is materialized on GPU
                # only one wrapped parameter at a time.
                stats = torch.zeros(3, dtype=torch.float64, device=device)
                stat_chunk_numel = 16 * 1024 * 1024  # at most 64 MiB fp32 CPU->GPU scratch
                for reward_shard, parameter in zip(reward_gradient_shards, parameters):
                    opd_shard = parameter.grad
                    if reward_shard is None and opd_shard is None:
                        continue
                    if reward_shard is not None and opd_shard is not None and reward_shard.shape != opd_shard.shape:
                        raise RuntimeError(
                            "C_par reward/OPD gradient shard shapes changed between backwards: "
                            f"{tuple(reward_shard.shape)} != {tuple(opd_shard.shape)}."
                        )
                    reward_flat = reward_shard.reshape(-1) if reward_shard is not None else None
                    opd_flat = opd_shard.detach().reshape(-1) if opd_shard is not None else None
                    numel = reward_flat.numel() if reward_flat is not None else opd_flat.numel()
                    for start in range(0, numel, stat_chunk_numel):
                        stop = min(start + stat_chunk_numel, numel)
                        reward_device = (
                            reward_flat[start:stop].to(device=device, dtype=torch.float32)
                            if reward_flat is not None
                            else None
                        )
                        opd_device = opd_flat[start:stop].float() if opd_flat is not None else None
                        if reward_device is not None:
                            stats[1] += torch.sum(reward_device.square(), dtype=torch.float64)
                        if opd_device is not None:
                            stats[2] += torch.sum(opd_device.square(), dtype=torch.float64)
                        if reward_device is not None and opd_device is not None:
                            stats[0] += torch.sum(reward_device * opd_device, dtype=torch.float64)
                        del reward_device, opd_device
                    del reward_flat, opd_flat

                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                dot = float(stats[0].item())
                reward_norm = float(torch.sqrt(stats[1].clamp_min(0.0)).item())
                opd_norm = float(torch.sqrt(stats[2].clamp_min(0.0)).item())
                norm_product = reward_norm * opd_norm
                cosine = dot / norm_product if norm_product > 0.0 else 0.0
                directional[reward_fold] = (dot, cosine, reward_norm, opd_norm)

                del reward_gradient_shards, stats
                self.actor_optimizer.zero_grad(set_to_none=True)
        finally:
            # C_par is diagnostic only: no gradient, optimizer state, or scheduler
            # state is allowed to leak into the real actor update below.
            self.actor_optimizer.zero_grad(set_to_none=True)

        if symmetric:
            dot_01, cos_01, reward_norm_0, opd_norm_1 = directional[0]
            dot_10, cos_10, reward_norm_1, opd_norm_0 = directional[1]
            return {
                "c_par/dot": 0.5 * (dot_01 + dot_10),
                "c_par/cosine": 0.5 * (cos_01 + cos_10),
                "c_par/dot_01": dot_01,
                "c_par/dot_10": dot_10,
                "c_par/cosine_01": cos_01,
                "c_par/cosine_10": cos_10,
                "c_par/reward_grad_norm": 0.5 * (reward_norm_0 + reward_norm_1),
                "c_par/opd_grad_norm": 0.5 * (opd_norm_0 + opd_norm_1),
                "c_par/reward_fold": -1.0,
                "c_par/global_sequences_fold0": global_sequences[0],
                "c_par/global_sequences_fold1": global_sequences[1],
                "c_par/global_tokens_fold0": global_tokens[0],
                "c_par/global_tokens_fold1": global_tokens[1],
            }

        dot, cosine, reward_norm, opd_norm = directional[requested_fold]
        return {
            "c_par/dot": dot,
            "c_par/cosine": cosine,
            "c_par/reward_grad_norm": reward_norm,
            "c_par/opd_grad_norm": opd_norm,
            "c_par/reward_fold": float(requested_fold),
            "c_par/global_sequences_reward_fold": global_sequences[requested_fold],
            "c_par/global_tokens_opd_fold": global_tokens[1 - requested_fold],
        }

    @GPUMemoryLogger(role="dp actor rl teacher oracle", logger=logger)
    def _measure_rl_teacher_alignment(self, data: DataProto, temperature: float) -> dict[str, float]:
        """Compare the training-teacher OPD direction with an RL residual oracle.

        All three gradients use the same on-policy student trajectories and the
        same student top-k support.  If ``base`` is the pre-RL student checkpoint,
        linearity of the stop-gradient OPD objective makes

            grad(L_RL) - grad(L_base)

        the finite RL-teacher residual direction.  Loss gradients and update
        directions share the same cosine because both are multiplied by -1.
        The diagnostic runs before the real optimizer step and clears every
        temporary gradient afterwards.
        """
        dist = torch.distributed
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("RL teacher alignment requires an initialized distributed process group.")
        if not isinstance(self.actor_module, FSDP):
            raise NotImplementedError("RL teacher alignment currently supports FSDP1 FULL_SHARD only.")
        if self.actor_module.sharding_strategy != ShardingStrategy.FULL_SHARD:
            raise NotImplementedError("RL teacher alignment requires fsdp_size=-1 / FULL_SHARD.")
        if int(self.ulysses_sequence_parallel_size) != 1:
            raise NotImplementedError("RL teacher alignment requires ulysses_sequence_parallel_size=1.")
        if str(self.config.loss_agg_mode) != "token-mean":
            raise NotImplementedError("RL teacher alignment currently requires loss_agg_mode=token-mean.")
        if "multi_modal_inputs" in data.non_tensor_batch:
            raise NotImplementedError("RL teacher alignment does not support multimodal batches.")

        required_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "student_top_k_ids",
            "student_top_k_log_probs",
            "big_teacher_on_student_log_probs",
            "big_native_teacher_on_student_log_probs",
            "rl_teacher_on_student_log_probs",
            "base_teacher_on_student_log_probs",
        ]
        missing = [key for key in required_keys if key not in data.batch]
        if missing:
            raise KeyError(f"RL teacher alignment input is missing tensor keys: {missing}.")
        optional_keys = ["format_mask"]
        selected = data.select(
            batch_keys=required_keys + [key for key in optional_keys if key in data.batch]
        )

        device = torch.device(self.device_name, get_device_id())
        world_size = dist.get_world_size()
        effective_mask = selected.batch["response_mask"].float()
        if "format_mask" in selected.batch:
            effective_mask = effective_mask * selected.batch["format_mask"].float().unsqueeze(-1)
        global_tokens_tensor = effective_mask.sum().to(device=device, dtype=torch.float64)
        dist.all_reduce(global_tokens_tensor, op=dist.ReduceOp.SUM)
        global_tokens = float(global_tokens_tensor.item())
        if global_tokens <= 0.0:
            raise ValueError("RL teacher alignment found no globally valid response tokens.")

        named_parameters = [
            (name, parameter)
            for name, parameter in self.actor_module.named_parameters()
            if parameter.requires_grad
        ]
        parameters = [parameter for _, parameter in named_parameters]
        if not parameters:
            raise RuntimeError("RL teacher alignment found no trainable actor parameters.")

        def make_micro_batches():
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(selected, max_token_len=max_token_len)
                return micro_batches
            return selected.split(int(self.config.ppo_micro_batch_size_per_gpu))

        # FSDP averages rank gradients.  Multiplying local token sums by
        # world_size/global_tokens recovers the global token-mean objective.
        loss_scale = float(world_size) / global_tokens

        def backward_teacher(
            teacher_key: str,
            capture_cpu: bool,
            reference_teacher_key: str | None = None,
            center_on_support: bool = False,
        ):
            self.actor_optimizer.zero_grad(set_to_none=True)
            finite_debug = os.getenv("GRAD_ORACLE_FINITE_DEBUG", "0") == "1"
            input_nonfinite: dict[str, int] = {}
            value_stats: dict[str, float] = {}

            def record_nonfinite(name: str, tensor: torch.Tensor, valid_mask: torch.Tensor) -> None:
                if not finite_debug:
                    return
                nonfinite = ~torch.isfinite(tensor.detach())
                expanded_valid = valid_mask.expand_as(tensor)
                input_nonfinite[f"{name}_valid"] = input_nonfinite.get(f"{name}_valid", 0) + int(
                    (nonfinite & expanded_valid).sum().item()
                )
                input_nonfinite[f"{name}_masked"] = input_nonfinite.get(f"{name}_masked", 0) + int(
                    (nonfinite & ~expanded_valid).sum().item()
                )

            def record_value_stats(name: str, tensor: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
                if not finite_debug:
                    return
                values = tensor.detach().float()
                if valid_mask is not None:
                    values = values[valid_mask.expand_as(values)]
                else:
                    values = values.reshape(-1)
                values = values[torch.isfinite(values)]
                if values.numel() == 0:
                    return
                current_min = float(values.min().item())
                current_max = float(values.max().item())
                current_absmax = float(values.abs().max().item())
                value_stats[f"{name}_min"] = min(value_stats.get(f"{name}_min", current_min), current_min)
                value_stats[f"{name}_max"] = max(value_stats.get(f"{name}_max", current_max), current_max)
                value_stats[f"{name}_absmax"] = max(
                    value_stats.get(f"{name}_absmax", current_absmax), current_absmax
                )

            for micro_batch in make_micro_batches():
                micro_batch = micro_batch.to(device)
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                loss_mask = model_inputs["response_mask"].float()
                if "format_mask" in model_inputs:
                    loss_mask = loss_mask * model_inputs["format_mask"].float().unsqueeze(-1)
                valid_support = loss_mask.bool().unsqueeze(-1)
                student_log_probs = model_inputs["student_top_k_log_probs"].float().detach()
                teacher_log_probs = model_inputs[teacher_key].float().detach()
                if student_log_probs.shape != teacher_log_probs.shape:
                    raise ValueError(
                        f"Student/teacher support shapes differ for {teacher_key}: "
                        f"{tuple(student_log_probs.shape)} != {tuple(teacher_log_probs.shape)}."
                    )
                record_nonfinite("student_log_probs", student_log_probs, valid_support)
                record_nonfinite("teacher_log_probs", teacher_log_probs, valid_support)
                record_value_stats("student_log_probs", student_log_probs, valid_support)
                record_value_stats("teacher_log_probs", teacher_log_probs, valid_support)
                # Padding/masked support entries are outside the objective.  Make
                # them explicitly finite before softmax/subtraction: multiplying
                # a later NaN by a zero loss mask would still poison autograd.
                student_log_probs = torch.where(
                    valid_support, student_log_probs, torch.zeros_like(student_log_probs)
                )
                teacher_log_probs = torch.where(
                    valid_support, teacher_log_probs, torch.zeros_like(teacher_log_probs)
                )
                support_weights = torch.softmax(student_log_probs, dim=-1)
                if reference_teacher_key is None:
                    log_ratio = teacher_log_probs - student_log_probs
                else:
                    reference_log_probs = model_inputs[reference_teacher_key].float().detach()
                    if reference_log_probs.shape != teacher_log_probs.shape:
                        raise ValueError(
                            f"Teacher/reference support shapes differ: {teacher_key}="
                            f"{tuple(teacher_log_probs.shape)} vs {reference_teacher_key}="
                            f"{tuple(reference_log_probs.shape)}."
                        )
                    record_nonfinite("reference_log_probs", reference_log_probs, valid_support)
                    reference_log_probs = torch.where(
                        valid_support, reference_log_probs, torch.zeros_like(reference_log_probs)
                    )
                    log_ratio = teacher_log_probs - reference_log_probs
                if center_on_support:
                    # With a full action expectation, any state-only additive
                    # constant cancels by the score-function identity. A
                    # renormalized top-k support breaks that identity, so remove
                    # the q_K-weighted state baseline explicitly. This makes the
                    # truncated oracle invariant to the RL policy's log Z(s).
                    state_baseline = (support_weights * log_ratio).sum(dim=-1, keepdim=True)
                    log_ratio = log_ratio - state_baseline
                advantages = log_ratio * support_weights
                record_value_stats("support_weights", support_weights, valid_support)
                record_value_stats("log_ratio", log_ratio, valid_support)
                record_value_stats("advantages", advantages, valid_support)
                target_ids = model_inputs["student_top_k_ids"]
                anomaly_context = (
                    torch.autograd.detect_anomaly(check_nan=True)
                    if os.getenv("GRAD_ORACLE_DETECT_ANOMALY", "0") == "1"
                    else contextlib.nullcontext()
                )
                with anomaly_context:
                    _, _, _, current_log_probs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        top_k=int(target_ids.shape[-1]),
                        student_top_k_ids=target_ids,
                    )
                    current_log_probs = current_log_probs.float()
                    record_nonfinite("current_log_probs", current_log_probs, valid_support)
                    record_value_stats("current_log_probs", current_log_probs, valid_support)
                    current_log_probs = torch.where(
                        valid_support, current_log_probs, torch.zeros_like(current_log_probs)
                    )
                    if finite_debug:
                        current_log_probs.retain_grad()
                    token_loss = -(advantages * current_log_probs).sum(dim=-1)
                    scaled_loss = (token_loss * loss_mask).sum().mul(loss_scale)
                    record_value_stats("token_loss", token_loss, loss_mask.bool())
                    record_value_stats("scaled_loss", scaled_loss)
                    if finite_debug:
                        input_nonfinite["scaled_loss"] = input_nonfinite.get("scaled_loss", 0) + int(
                            (~torch.isfinite(scaled_loss.detach())).sum().item()
                        )
                    scaled_loss.backward()
                    if finite_debug and current_log_probs.grad is not None:
                        record_nonfinite("current_log_probs_grad", current_log_probs.grad, valid_support)
                        record_value_stats("current_log_probs_grad", current_log_probs.grad, valid_support)

            if finite_debug:
                bad_parameter_summaries = []
                nonfinite_gradient_elements = 0
                gradient_elements = 0
                for name, parameter in named_parameters:
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad.detach()
                    gradient_elements += gradient.numel()
                    bad_count = int((~torch.isfinite(gradient)).sum().item())
                    nonfinite_gradient_elements += bad_count
                    if bad_count and len(bad_parameter_summaries) < 12:
                        bad_parameter_summaries.append(
                            {"name": name, "bad": bad_count, "numel": gradient.numel(), "dtype": str(gradient.dtype)}
                        )
                local_summary = {
                    "rank": dist.get_rank(),
                    "teacher": teacher_key,
                    "input_nonfinite": input_nonfinite,
                    "value_stats": value_stats,
                    "gradient_elements": gradient_elements,
                    "nonfinite_gradient_elements": nonfinite_gradient_elements,
                    "bad_parameters": bad_parameter_summaries,
                }
                gathered_summaries = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_summaries, local_summary)
                if dist.get_rank() == 0:
                    print(f"[grad-oracle-finite-debug] {gathered_summaries}", flush=True)
                valid_input_bad = any(
                    value > 0
                    for summary in gathered_summaries
                    for name, value in summary["input_nonfinite"].items()
                    if name.endswith("_valid") or name == "scaled_loss"
                )
                gradient_bad = any(
                    summary["nonfinite_gradient_elements"] > 0 for summary in gathered_summaries
                )
                if valid_input_bad or gradient_bad:
                    raise FloatingPointError(
                        f"Non-finite RL-oracle backward for {teacher_key}; "
                        "see [grad-oracle-finite-debug] rank summaries."
                    )

            if not capture_cpu:
                return None
            return [
                None
                if parameter.grad is None
                else parameter.grad.detach().to(device="cpu", dtype=torch.float32, copy=True)
                for parameter in parameters
            ]

        self.actor_optimizer.zero_grad(set_to_none=True)
        try:
            big_gradient_shards = backward_teacher("big_teacher_on_student_log_probs", capture_cpu=True)
            native_big_gradient_shards = backward_teacher(
                "big_native_teacher_on_student_log_probs", capture_cpu=True
            )
            base_gradient_shards = backward_teacher("base_teacher_on_student_log_probs", capture_cpu=True)
            centered_residual_gradient_shards = backward_teacher(
                "rl_teacher_on_student_log_probs",
                capture_cpu=True,
                reference_teacher_key="base_teacher_on_student_log_probs",
                center_on_support=True,
            )
            backward_teacher("rl_teacher_on_student_log_probs", capture_cpu=False)

            # [big·residual, ||big||², ||residual||²,
            #  big·rl, ||rl||², big·base, ||base||², rl·base]
            # Additional slots: native_big·residual, ||native_big||²,
            # raw_big·native_big.  This isolates the four-token native-template
            # difference discovered by the tokenizer preflight.
            # Final slots: raw_big·centered_residual,
            # native_big·centered_residual, ||centered_residual||².
            stats = torch.zeros(14, dtype=torch.float64, device=device)
            stat_chunk_numel = 16 * 1024 * 1024
            for big_shard, native_big_shard, base_shard, centered_residual_shard, parameter in zip(
                big_gradient_shards,
                native_big_gradient_shards,
                base_gradient_shards,
                centered_residual_gradient_shards,
                parameters,
            ):
                rl_shard = parameter.grad
                reference_shape = next(
                    (
                        tensor.shape
                        for tensor in (
                            big_shard,
                            native_big_shard,
                            base_shard,
                            centered_residual_shard,
                            rl_shard,
                        )
                        if tensor is not None
                    ),
                    None,
                )
                if reference_shape is None:
                    continue
                if any(
                    tensor is not None and tensor.shape != reference_shape
                    for tensor in (
                        big_shard,
                        native_big_shard,
                        base_shard,
                        centered_residual_shard,
                        rl_shard,
                    )
                ):
                    raise RuntimeError("RL teacher diagnostic gradient shard shapes changed between backwards.")
                numel = int(torch.tensor(reference_shape).prod().item())
                big_flat = big_shard.reshape(-1) if big_shard is not None else None
                native_big_flat = native_big_shard.reshape(-1) if native_big_shard is not None else None
                base_flat = base_shard.reshape(-1) if base_shard is not None else None
                centered_residual_flat = (
                    centered_residual_shard.reshape(-1)
                    if centered_residual_shard is not None
                    else None
                )
                rl_flat = rl_shard.detach().reshape(-1) if rl_shard is not None else None
                for start in range(0, numel, stat_chunk_numel):
                    stop = min(start + stat_chunk_numel, numel)
                    big = (
                        big_flat[start:stop].to(device=device, dtype=torch.float32)
                        if big_flat is not None
                        else torch.zeros(stop - start, device=device, dtype=torch.float32)
                    )
                    base = (
                        base_flat[start:stop].to(device=device, dtype=torch.float32)
                        if base_flat is not None
                        else torch.zeros(stop - start, device=device, dtype=torch.float32)
                    )
                    native_big = (
                        native_big_flat[start:stop].to(device=device, dtype=torch.float32)
                        if native_big_flat is not None
                        else torch.zeros(stop - start, device=device, dtype=torch.float32)
                    )
                    rl = (
                        rl_flat[start:stop].float()
                        if rl_flat is not None
                        else torch.zeros(stop - start, device=device, dtype=torch.float32)
                    )
                    centered_residual = (
                        centered_residual_flat[start:stop].to(device=device, dtype=torch.float32)
                        if centered_residual_flat is not None
                        else torch.zeros(stop - start, device=device, dtype=torch.float32)
                    )
                    residual = rl - base
                    stats[0] += torch.sum(big * residual, dtype=torch.float64)
                    stats[1] += torch.sum(big.square(), dtype=torch.float64)
                    stats[2] += torch.sum(residual.square(), dtype=torch.float64)
                    stats[3] += torch.sum(big * rl, dtype=torch.float64)
                    stats[4] += torch.sum(rl.square(), dtype=torch.float64)
                    stats[5] += torch.sum(big * base, dtype=torch.float64)
                    stats[6] += torch.sum(base.square(), dtype=torch.float64)
                    stats[7] += torch.sum(rl * base, dtype=torch.float64)
                    stats[8] += torch.sum(native_big * residual, dtype=torch.float64)
                    stats[9] += torch.sum(native_big.square(), dtype=torch.float64)
                    stats[10] += torch.sum(big * native_big, dtype=torch.float64)
                    stats[11] += torch.sum(big * centered_residual, dtype=torch.float64)
                    stats[12] += torch.sum(native_big * centered_residual, dtype=torch.float64)
                    stats[13] += torch.sum(centered_residual.square(), dtype=torch.float64)
                    del big, native_big, base, rl, residual, centered_residual

            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            dot_big_residual = float(stats[0].item())
            big_norm = float(torch.sqrt(stats[1].clamp_min(0.0)).item())
            residual_norm = float(torch.sqrt(stats[2].clamp_min(0.0)).item())
            dot_big_rl = float(stats[3].item())
            rl_norm = float(torch.sqrt(stats[4].clamp_min(0.0)).item())
            dot_big_base = float(stats[5].item())
            base_norm = float(torch.sqrt(stats[6].clamp_min(0.0)).item())
            dot_rl_base = float(stats[7].item())
            dot_native_big_residual = float(stats[8].item())
            native_big_norm = float(torch.sqrt(stats[9].clamp_min(0.0)).item())
            dot_big_native_big = float(stats[10].item())
            dot_big_centered_residual = float(stats[11].item())
            dot_native_big_centered_residual = float(stats[12].item())
            centered_residual_norm = float(torch.sqrt(stats[13].clamp_min(0.0)).item())

            def cosine(dot: float, norm_a: float, norm_b: float) -> float:
                denominator = norm_a * norm_b
                return dot / denominator if denominator > 0.0 else 0.0

            residual_norm_sq = residual_norm * residual_norm
            return {
                "grad_oracle/cos_big_rl_residual": cosine(dot_big_residual, big_norm, residual_norm),
                "grad_oracle/cos_big_native_rl_residual": cosine(
                    dot_native_big_residual, native_big_norm, residual_norm
                ),
                "grad_oracle/cos_big_rl_residual_centered": cosine(
                    dot_big_centered_residual, big_norm, centered_residual_norm
                ),
                "grad_oracle/cos_big_native_rl_residual_centered": cosine(
                    dot_native_big_centered_residual, native_big_norm, centered_residual_norm
                ),
                "grad_oracle/cos_big_raw_native": cosine(
                    dot_big_native_big, big_norm, native_big_norm
                ),
                "grad_oracle/cos_big_rl_full": cosine(dot_big_rl, big_norm, rl_norm),
                "grad_oracle/cos_big_base": cosine(dot_big_base, big_norm, base_norm),
                "grad_oracle/cos_rl_base": cosine(dot_rl_base, rl_norm, base_norm),
                "grad_oracle/dot_big_rl_residual": dot_big_residual,
                "grad_oracle/dot_big_native_rl_residual": dot_native_big_residual,
                "grad_oracle/dot_big_rl_residual_centered": dot_big_centered_residual,
                "grad_oracle/dot_big_native_rl_residual_centered": dot_native_big_centered_residual,
                "grad_oracle/big_grad_norm": big_norm,
                "grad_oracle/big_native_grad_norm": native_big_norm,
                "grad_oracle/rl_residual_grad_norm": residual_norm,
                "grad_oracle/rl_residual_centered_grad_norm": centered_residual_norm,
                "grad_oracle/rl_full_grad_norm": rl_norm,
                "grad_oracle/base_grad_norm": base_norm,
                "grad_oracle/big_projection_on_rl_residual": (
                    dot_big_residual / residual_norm_sq if residual_norm_sq > 0.0 else 0.0
                ),
                "grad_oracle/big_native_projection_on_rl_residual": (
                    dot_native_big_residual / residual_norm_sq if residual_norm_sq > 0.0 else 0.0
                ),
                "grad_oracle/big_projection_on_rl_residual_centered": (
                    dot_big_centered_residual / (centered_residual_norm * centered_residual_norm)
                    if centered_residual_norm > 0.0
                    else 0.0
                ),
                "grad_oracle/big_native_projection_on_rl_residual_centered": (
                    dot_native_big_centered_residual / (centered_residual_norm * centered_residual_norm)
                    if centered_residual_norm > 0.0
                    else 0.0
                ),
                "grad_oracle/global_tokens": global_tokens,
            }
        finally:
            self.actor_optimizer.zero_grad(set_to_none=True)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        metrics = {}

        diagnostic_keys = (
            "big_native_teacher_on_student_log_probs",
            "rl_teacher_on_student_log_probs",
            "base_teacher_on_student_log_probs",
        )
        if all(key in data.batch for key in diagnostic_keys):
            oracle_metrics = self._measure_rl_teacher_alignment(data=data, temperature=temperature)
            append_to_dict(metrics, oracle_metrics)

        # Run on the untouched local DP shard and before *any* optimizer step.
        # The driver only attaches these fields on configured measurement steps.
        if "c_par_fold" in data.batch and "c_par_reward_advantages" in data.batch:
            c_par_metrics = self._measure_c_par(data=data, temperature=temperature)
            append_to_dict(metrics, c_par_metrics)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # Check if we have 3D advantages (top-k sampling case)
                    # If so, we need to recompute top-k log probs for correct gradient
                    if advantages.dim() == 3:
                        top_k = advantages.shape[-1]
                        # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        entropy, _, _, topk_log_probs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            top_k=top_k, student_top_k_ids=student_top_k_ids
                        )
                        log_prob_for_loss = topk_log_probs
                        
                    else:
                        _, log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        log_prob_for_loss = log_prob

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
            

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            print("on_policy")
                            # For on-policy (ppo_epochs=1), use current policy as "old"
                            # log_prob_for_loss is already 3D for top-k case
                            old_log_prob = log_prob_for_loss.detach()
                        else:
                            print("off_policy")
                            # For off-policy, use stored log probs
                            # For 3D top-k case, use stored log probs (union or student)
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        format_mask=format_mask,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

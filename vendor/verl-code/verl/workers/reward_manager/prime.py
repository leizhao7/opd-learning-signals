# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(executor, partial(evaluation_func, task, completion, reference, task_extra_info))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the
        # exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=300.0)
                for c, r, t, ei in zip(completions, references, tasks, extra_info, strict=True)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks, strict=True):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result, int | float | bool):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores


# ---------------------------------------------------------------------------------------------
# 2026-09-23 补丁：原实现用 ProcessPoolExecutor，打完分在 finally 里强杀 worker、再在 with 退出时
# shutdown(wait=True) 等管理线程，偶发永久死锁（opd_code_rl4b_rerun step 50 验证后挂死 50 分钟），
# 且被杀 worker 的孙进程过继给容器 1 号进程成为僵尸。现改为独立会话里的隔离打分子进程，
# 见 _prime_isolated_runner.py。并发数、FIFO 顺序、整批 300 秒截止、分数换算规则均与原版一致。
# 原函数 single_compute_score / parallel_compute_score_async 保留未删，仅不再被调用。
# ---------------------------------------------------------------------------------------------
_ISO_IMPORT_ALLOWANCE_S = 180.0  # 隔离子进程 import 打分模块的时间上限（不计入 300 秒打分预算）
_ISO_GRACE_S = 60.0
print("[prime-isolated] patched reward scoring active (v2: resident workers w/ private pipes, process groups, subreaper, outer guard)", flush=True)


def _kill_session(sid):
    import signal as _signal

    n = 0
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            if os.getsid(pid) == sid:
                os.kill(pid, _signal.SIGKILL)
                n += 1
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return n


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64, timeout=300.0):
    import json
    import pickle
    import subprocess
    import sys
    import tempfile
    import time

    n = len(tasks)
    if extra_info is None:
        extra_info = [None] * n
    comps, refs, tks, exs = list(completions), list(references), list(tasks), list(extra_info)
    assert len(comps) == len(refs) == len(tks) == len(exs) == n
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_prime_isolated_runner.py")
    t0 = time.monotonic()
    results = [None] * n
    summary = None
    with tempfile.TemporaryDirectory(prefix="prime_iso_") as td:
        inp, out = os.path.join(td, "in.pkl"), os.path.join(td, "out.pkl")
        with open(inp, "wb") as f:
            pickle.dump(dict(func=evaluation_func, completions=comps, references=refs, tasks=tks, extra_info=exs,
                             num_processes=num_processes, timeout=timeout), f)
        env = dict(os.environ, PRIME_ISO_SYS_PATH=json.dumps([p for p in sys.path if p]))
        if os.environ.get("PRIME_ISO_SINGLE_THREAD", "0") == "1":  # 默认不改线程环境，与原版一致
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        proc = subprocess.Popen([sys.executable, "-u", runner, inp, out], start_new_session=True, env=env)
        try:
            proc.wait(timeout=timeout + _ISO_IMPORT_ALLOWANCE_S + _ISO_GRACE_S)
        except subprocess.TimeoutExpired:
            print(f"[Shutdown] isolated scorer exceeded outer guard; killing its session", flush=True)
            _kill_session(proc.pid)
            proc.wait()
        leftover = _kill_session(proc.pid)  # 保险：会话内不留任何进程
        if os.path.exists(out):
            with open(out, "rb") as f:
                payload = pickle.load(f)
            results, summary = payload["results"], payload["summary"]
    scores = [0.0 if r is None else float(r) for r in results]
    print(f"[Shutdown] isolated scoring: {summary} leftover_killed={leftover} total_s={time.monotonic() - t0:.1f}", flush=True)
    return scores


@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=64,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # thunlp/OPD naive-manager pattern: ALWAYS compute the rule-based outcome
        # below; when distillation rm_scores are present they stay the training
        # reward, and the real outcome goes out as true_reward_score.

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str[i])

        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                reward_extra_info["true_reward_score"] = reward_tensor
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor

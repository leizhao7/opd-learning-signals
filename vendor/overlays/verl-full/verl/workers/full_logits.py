"""Exact full-support on-policy OPD, chunked over states (never vocabulary).

Only the tested G=1, one-minibatch, one-epoch, student_p reverse-OPD case
is supported. Colocated reward/actor workers cache frozen hidden states,
not B x T x V arrays. The existing dynamic microbatch partitions, their
loss scaling, optimizer, scheduler, rollout and evaluation remain unchanged.
"""
import contextlib
import hashlib
import types
import torch
import torch.nn.functional as F

VOCAB = 151936
CHUNK = 128
CACHE = {}


def enabled(k):
    return int(k) == VOCAB


def amp(t):
    return torch.autocast('cuda', dtype=torch.bfloat16) if t.is_cuda else contextlib.nullcontext()


def projection_dtype(t):
    return torch.bfloat16 if t.is_cuda else t.dtype


def lp(logits, temperature):
    # Match the original autocast + in-place temperature + log_softmax path.
    with amp(logits):
        return torch.log_softmax(logits / temperature, dim=-1)


def project(h, w):
    with amp(h):
        return F.linear(h.to(w.dtype), w)


def frozen_advantage(old_h, old_w, teacher_h, teacher_w, st, tt):
    slp = lp(project(old_h, old_w), st)
    tlp = lp(project(teacher_h, teacher_w), tt)
    assert tlp.shape[-1] >= slp.shape[-1]
    # Normalize over the full teacher vocabulary before aligning student IDs.
    tlp = tlp[..., :slp.shape[-1]]
    weights = (slp - torch.logsumexp(slp, dim=-1, keepdim=True)).exp()
    return -(slp - tlp) * weights


class FullSupportSurrogate(torch.autograd.Function):
    """Same on-policy PPO value/gradient, without retaining full-vocab arrays.

    For the existing on-policy branch, old_log_prob=current.detach(), ratio=1,
    and all PPO clipping branches coincide. Advantages remain frozen from the
    pre-update student and teacher, exactly as in the top-k implementation.
    """
    @staticmethod
    def forward(ctx, hidden, weight, old_h, old_w, teacher_h, teacher_w,
                student_temperature, teacher_temperature, chunk_size=CHUNK):
        n = hidden.shape[0]
        assert n > 0 and weight.shape[0] == old_w.shape[0] and teacher_w.shape[0] >= weight.shape[0]
        dtype = projection_dtype(hidden)
        w = weight.detach().to(dtype)
        ow, tw = old_w.to(dtype), teacher_w.to(dtype)
        dh = torch.empty_like(hidden)
        dw = torch.zeros_like(weight, dtype=torch.float64 if weight.dtype == torch.float64 else torch.float32)
        value = torch.zeros((), device=hidden.device, dtype=torch.float64 if hidden.dtype == torch.float64 else torch.float32)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            h = hidden[start:stop].detach().to(dtype)
            adv = frozen_advantage(old_h[start:stop].to(dtype), ow,
                                   teacher_h[start:stop].to(dtype), tw,
                                   student_temperature, teacher_temperature)
            logits = project(h, w)
            # Let PyTorch compute the exact log-softmax gradient, including
            # the same mixed-precision casts as the dense reference.
            with torch.enable_grad():
                logits = logits.detach().requires_grad_(True)
                current = lp(logits, student_temperature)
                ratio = (current - current.detach()).exp()
                part = (-adv.detach() * ratio).sum() / n
                dz, = torch.autograd.grad(part, logits)
            value += part.detach().to(value.dtype)
            with amp(hidden):
                dh[start:stop] = torch.mm(dz.to(dtype), w).to(hidden.dtype)
                dw.add_(torch.mm(dz.to(dtype).T, h).to(dw.dtype))
        ctx.save_for_backward(dh, dw.to(weight.dtype))
        return value

    @staticmethod
    def backward(ctx, grad_output):
        dh, dw = ctx.saved_tensors
        return (dh * grad_output, dw * grad_output, None, None, None, None,
                None, None, None)


def keys(batch):
    result = []
    for i in range(batch['input_ids'].shape[0]):
        h = hashlib.sha256()
        for name in ['input_ids', 'attention_mask', 'position_ids']:
            h.update(batch[name][i].detach().cpu().contiguous().numpy().tobytes())
        result.append(h.hexdigest())
    return result


def install_forward(module):
    raw = getattr(module, '_fsdp_wrapped_module', module)
    if getattr(raw, '_full_logits_forward_installed', False):
        return
    assert raw.config.model_type in ('qwen2', 'qwen3') and raw.config.vocab_size in (VOCAB, 152064)
    assert raw.lm_head.bias is None
    original = raw.forward

    def forward(self, *args, full_logits_role=None, full_logits_payload=None, **kwargs):
        if full_logits_role is None:
            return original(*args, **kwargs)
        assert not args
        kwargs.pop('return_dict', None)
        output = self.model(**kwargs, return_dict=True)
        hidden = output.last_hidden_state.squeeze(0)
        if full_logits_role in ('student', 'teacher'):
            name = full_logits_role + '_weight'
            if name not in CACHE:
                CACHE[name] = self.lm_head.weight.detach().to(projection_dtype(hidden)).clone()
            return hidden.detach().to(projection_dtype(hidden))
        assert full_logits_role == 'update'
        p = full_logits_payload
        selected = hidden.index_select(0, p['indices'])
        return FullSupportSurrogate.apply(selected, self.lm_head.weight,
                                         p['old_h'], CACHE['student_weight'],
                                         p['teacher_h'], CACHE['teacher_weight'],
                                         p['student_temperature'], p['teacher_temperature'], CHUNK)

    raw.forward = types.MethodType(forward, raw)
    raw._full_logits_forward_installed = True


def packed_inputs(batch):
    from verl.utils.attention_utils import unpad_input, index_first_axis
    x = batch['input_ids']
    assert batch['position_ids'].ndim == 2
    values, indices, *_ = unpad_input(x.unsqueeze(-1), batch['attention_mask'])
    positions = index_first_axis(batch['position_ids'].reshape(-1, 1), indices).T
    kwargs = {'input_ids': values.T, 'attention_mask': None,
              'position_ids': positions, 'use_cache': False}
    return kwargs, indices


def extract_response_hidden(module, batch, role):
    from verl.utils.attention_utils import pad_input
    install_forward(module)
    kwargs, indices = packed_inputs(batch)
    with torch.no_grad(), amp(batch['input_ids']):
        hidden = module(**kwargs, full_logits_role=role)
    b, length = batch['input_ids'].shape
    full = pad_input(hidden, indices, batch=b, seqlen=length)
    response_len = batch['responses'].shape[1]
    return full[:, -response_len-1:-1].contiguous()


def cache_batch(role, module, batch):
    hidden = extract_response_hidden(module, batch, role)
    entries = CACHE.setdefault(role, {})
    for key, h in zip(keys(batch), hidden):
        entries[key] = h.detach().clone()
    return hidden


def sampled_statistics(hidden, weight, batch, temperature):
    b, length, width = hidden.shape
    valid = batch['attention_mask'][:, -length-1:-1].bool().reshape(-1)
    indices = valid.nonzero().squeeze(-1)
    labels = batch['responses'].reshape(-1).index_select(0, indices)
    hs = hidden.reshape(-1, width).index_select(0, indices)
    logs = torch.zeros(b*length, device=hidden.device, dtype=torch.float32)
    entropy = torch.zeros_like(logs)
    with torch.no_grad():
        for start in range(0, len(indices), CHUNK):
            stop = min(start+CHUNK, len(indices))
            logp = lp(project(hs[start:stop], weight), temperature)
            logs[indices[start:stop]] = logp.gather(-1, labels[start:stop, None]).squeeze(-1).float()
            entropy[indices[start:stop]] = -(logp.exp()*logp).sum(-1).float()
    return logs.reshape(b, length), entropy.reshape(b, length)


def actor_log_probs(actor, data):
    from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
    from verl.utils.device import get_device_id
    assert actor.use_remove_padding and actor.ulysses_sequence_parallel_size == 1
    CACHE.clear()
    actor.actor_module.eval()
    selected = data.select(batch_keys=['responses','input_ids','attention_mask','position_ids'])
    dynamic = data.meta_info['use_dynamic_bsz']
    if dynamic:
        batches, order = prepare_dynamic_batch(selected, max_token_len=data.meta_info['max_token_len'])
    else:
        batches = selected.split(data.meta_info['micro_batch_size'])
    logs, entropies = [], []
    for mb in batches:
        batch = mb.to(get_device_id()).batch
        hidden = cache_batch('student', actor.actor_module, batch)
        logp, ent = sampled_statistics(hidden, CACHE['student_weight'], batch, data.meta_info['temperature'])
        logs.append(logp); entropies.append(ent)
    logs, entropies = torch.cat(logs), torch.cat(entropies)
    if dynamic:
        logs = restore_dynamic_batch(logs, order)
        entropies = restore_dynamic_batch(entropies, order)
    print('FULL_LOGITS_STUDENT_CACHED', len(CACHE['student']), VOCAB, flush=True)
    return logs, entropies, None, None


def teacher_scores(worker, data):
    from verl import DataProto
    from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
    from verl.utils.device import get_device_id
    assert worker.use_remove_padding and worker.ulysses_sequence_parallel_size == 1
    assert not worker._do_switch_chat_template
    assert 'student_weight' in CACHE, 'Actor and teacher must be colocated on the same DP rank'
    CACHE.pop('teacher', None); CACHE.pop('teacher_weight', None)
    worker.reward_module.eval()
    selected = data.select(batch_keys=['responses','input_ids','attention_mask','position_ids'])
    dynamic = worker.config.use_dynamic_bsz
    with worker.ulysses_sharding_manager:
        if dynamic:
            batches, order = prepare_dynamic_batch(selected, max_token_len=worker.config.forward_max_token_len_per_gpu)
        else:
            batches = selected.split(worker.config.micro_batch_size_per_gpu)
        entropy = []
        for mb in batches:
            batch = mb.to(get_device_id()).batch
            hidden = cache_batch('teacher', worker.reward_module, batch)
            _, ent = sampled_statistics(hidden, CACHE['teacher_weight'], batch, data.meta_info['teacher_temperature'])
            entropy.append(ent)
        entropy = torch.cat(entropy)
        if dynamic:entropy = restore_dynamic_batch(entropy, order)
    if worker.world_size > 1:worker.reward_module._handle.reshard(True)
    print('FULL_LOGITS_TEACHER_CACHED', len(CACHE['teacher']), CACHE['teacher_weight'].shape[0], flush=True)
    return DataProto.from_dict(tensors={'teacher_entropy':entropy})


def distillation_reward(actor, data):
    from verl import DataProto
    from verl.utils.device import get_device_id
    assert data.meta_info['top_k_strategy'] == 'only_stu'
    assert data.meta_info['reward_weight_mode'] == 'student_p'
    batch = data.to(get_device_id()).batch
    mask = batch['response_mask'].bool()
    rewards = torch.zeros((*mask.shape,1), device=mask.device, dtype=torch.float32)
    with torch.no_grad():
        for i, key in enumerate(keys(batch)):
            valid = mask[i]
            sh = CACHE['student'][key][valid]
            th = CACHE['teacher'][key][valid]
            sums=[]
            for start in range(0, len(sh), CHUNK):
                adv=frozen_advantage(sh[start:start+CHUNK], CACHE['student_weight'],
                                     th[start:start+CHUNK], CACHE['teacher_weight'],
                                     data.meta_info['temperature'], data.meta_info['teacher_temperature'])
                sums.append(adv.sum(-1).float())
            rewards[i,valid,0]=torch.cat(sums)
    assert torch.isfinite(rewards).all()
    # The final singleton contains the exact sum over all V actions, solely
    # for driver statistics. The update uses frozen hidden states, not this
    # compressed scalar as a sampled-action advantage.
    return DataProto.from_dict(tensors={'rm_scores':rewards})


def policy_loss(actor, batch, student_temperature, teacher_temperature):
    install_forward(actor.actor_module)
    kwargs, packed_indices = packed_inputs(batch)
    b, seqlen = batch['input_ids'].shape
    response_len = batch['responses'].shape[1]
    mask = batch['response_mask'].bool()
    inverse = torch.full((b*seqlen,), -1, dtype=torch.long, device=packed_indices.device)
    inverse[packed_indices] = torch.arange(len(packed_indices), device=packed_indices.device)
    full_positions = (torch.arange(b,device=mask.device)[:,None]*seqlen
                      + torch.arange(seqlen-response_len-1,seqlen-1,device=mask.device)[None,:])
    take = inverse[full_positions[mask]]
    assert (take >= 0).all()
    old_h = torch.cat([CACHE['student'][k][m] for k,m in zip(keys(batch),mask)])
    teacher_h = torch.cat([CACHE['teacher'][k][m] for k,m in zip(keys(batch),mask)])
    payload={'indices':take,'old_h':old_h,'teacher_h':teacher_h,
             'student_temperature':student_temperature,'teacher_temperature':teacher_temperature}
    with amp(batch['input_ids']):
        return actor.actor_module(**kwargs, full_logits_role='update', full_logits_payload=payload)


def update_policy(actor, data):
    from verl.utils.seqlen_balancing import prepare_dynamic_batch
    from verl.utils.device import get_device_id
    from verl.utils.py_functional import append_to_dict
    cfg=actor.config
    assert cfg.ppo_epochs==1 and cfg.rollout_n==1 and cfg.loss_agg_mode=='token-mean'
    assert not cfg.use_kl_loss and float(cfg.entropy_coeff)==0
    assert cfg.policy_loss.get('loss_mode','vanilla')=='vanilla'
    assert not cfg.c_par.enable and 'rollout_is_weights' not in data.batch and 'format_mask' not in data.batch
    assert not bool(getattr(cfg,'use_rollout_log_probs',False))
    assert data.meta_info['reward_weight_mode']=='student_p'
    assert data.meta_info['top_k_strategy']=='only_stu'
    actor.actor_module.train()
    selected=data.select(batch_keys=['responses','response_mask','input_ids','attention_mask','position_ids'])
    mini_batches=selected.split(cfg.ppo_mini_batch_size)
    assert len(mini_batches)==1, 'Full-support path requires the same on-policy single minibatch'
    metrics={}
    for mini_batch in mini_batches:
        if cfg.use_dynamic_bsz:
            batches,_=prepare_dynamic_batch(mini_batch,max_token_len=cfg.ppo_max_token_len_per_gpu)
        else:
            actor.gradient_accumulation=cfg.ppo_mini_batch_size//cfg.ppo_micro_batch_size_per_gpu
            batches=mini_batch.split(cfg.ppo_micro_batch_size_per_gpu)
        actor.actor_optimizer.zero_grad()
        for mb in batches:
            batch=mb.to(get_device_id()).batch
            scale=batch['response_mask'].shape[0]/cfg.ppo_mini_batch_size if cfg.use_dynamic_bsz else 1/actor.gradient_accumulation
            loss=policy_loss(actor,batch,data.meta_info['temperature'],data.meta_info['teacher_temperature'])
            assert torch.isfinite(loss), 'Non-finite full-support loss'
            (loss*scale).backward()
            append_to_dict(metrics,{'actor/pg_loss':loss.detach().item()*scale,
                                   'actor/pg_clipfrac':0.0,'actor/ppo_kl':0.0,'actor/pg_clipfrac_lower':0.0})
        grad_norm=actor._optimizer_step()
        append_to_dict(metrics,{'actor/grad_norm':grad_norm.detach().item(),
                               'full_logits/vocab_size':VOCAB,'full_logits/teacher_vocab_size':CACHE['teacher_weight'].shape[0],'full_logits/state_chunk_size':CHUNK})
    actor.actor_optimizer.zero_grad()
    CACHE.clear()
    return metrics

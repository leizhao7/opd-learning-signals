"""One legitimate teacher generation worker per GPU; append-only attempt provenance."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--rank', type=int, required=True)
    a = p.parse_args()
    job = json.loads(Path(a.job).read_text())
    assert job['authorized'] and job['config_resolved']
    rank = a.rank
    assert 0 <= rank < len(job['devices'])
    cfg = job['rollout']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(job['devices'][rank])
    out = Path(job['run_dir']) / 'rollout'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / f'worker{rank}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert digest(job['prompt_file']) == job['prompt_sha256']
    rows = [json.loads(l) for l in Path(job['prompt_file']).read_text().splitlines()]
    rows = rows[rank::len(job['devices'])]
    # Quantity means candidate slots, or accepted rows, explicitly fixed in the contract.
    quota = job['target_rows'] // len(job['devices'])
    assert job['target_rows'] % len(job['devices']) == 0
    accepted_path = out / f'accepted{rank}.jsonl'
    attempts_path = out / f'attempts{rank}.jsonl'
    accepted = [json.loads(l) for l in accepted_path.read_text().splitlines()] if accepted_path.exists() else []
    attempts = [json.loads(l) for l in attempts_path.read_text().splitlines()] if attempts_path.exists() else []
    done = {r['slot_id'] for r in accepted}
    counts = collections.Counter(r['slot_id'] for r in attempts)
    assert len(accepted) == len(done) and len(done) <= quota
    spec = importlib.util.spec_from_file_location('reference_rollout', job['reference_rollout'])
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    if len(done) == quota:
        write(out / f'worker{rank}_complete.json', {
            'accepted':len(done),'attempts':sum(counts.values()),
            'accepted_sha256':digest(accepted_path),'attempts_sha256':digest(attempts_path),
            'completed_unix':time.time(),'reconstructed_after_completed_data':True})
        return
    from vllm import LLM, SamplingParams
    import torch
    from transformers import AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer
    student_tok = AutoTokenizer.from_pretrained(job['student'],local_files_only=True,trust_remote_code=False)
    student_template = get_template_and_fix_tokenizer(student_tok,DataArguments(
        template=job['sft_config']['template'],enable_thinking=job['sft_config']['enable_thinking']))
    # GPU scheduling is handled by the caller; no cluster holder is used.
    llm = LLM(model=job['teacher'], tensor_parallel_size=1, dtype='bfloat16',
              max_model_len=cfg['max_model_len'], gpu_memory_utilization=0.85,
              trust_remote_code=False, seed=cfg['seed'] + rank)
    tok = llm.get_tokenizer()
    sampling = SamplingParams(n=1, temperature=cfg['temperature'], top_p=cfg['top_p'],
                              top_k=cfg['top_k'], max_tokens=cfg['max_tokens'],
                              repetition_penalty=1.0)
    write(out / f'worker{rank}_started.json', {
        'pid': os.getpid(), 'rank': rank, 'start_unix': time.time(),
        'config': cfg, 'teacher': job['teacher'], 'prompt_sha256': job['prompt_sha256'],
        'versions': {'vllm': __import__('vllm').__version__, 'torch': torch.__version__}})
    pending = [r for r in rows if r['slot_id'] not in done and counts[r['slot_id']] < cfg['max_attempts']]
    if job['quantity_mode'] == 'candidate_slots':
        permitted = {r['slot_id'] for r in rows[:quota]}
        pending = [r for r in pending if r['slot_id'] in permitted]
    with accepted_path.open('a') as good, attempts_path.open('a') as raw:
        while pending and (job['quantity_mode'] != 'accepted_rows' or len(done) < quota):
            n = min(cfg['generation_batch_size'], len(pending))
            if job['quantity_mode'] == 'accepted_rows':
                n = min(n, quota - len(done))
            batch, pending = pending[:n], pending[n:]
            prompts = [tok.apply_chat_template(r['messages'], tokenize=False,
                        add_generation_prompt=True, enable_thinking=job['teacher_enable_thinking']) for r in batch]
            generated = llm.generate(prompts, sampling, use_tqdm=False)
            assert len(generated) == len(batch), 'Generation count mismatch'
            retry = []
            for r, prompt, generation in zip(batch, prompts, generated):
                sample = generation.outputs[0]
                response = sample.text
                valid, reason = ref.is_valid_output(response)
                if sample.finish_reason != 'stop':
                    valid, reason = False, 'incomplete_generation'
                if job['model_family'] == 'r1' and '</think>' not in response:
                    valid, reason = False, 'unclosed_reasoning'
                target = response
                if job['model_family'] == 'r1' and not target.lstrip().startswith('<think>'):
                    target = '<think>\n' + target
                student_x, student_y = student_template.encode_oneturn(student_tok,
                    r['messages'] + [{'role':'assistant','content':target}])
                student_length = len(student_x) + len(student_y)
                if student_length > job['sft_config']['cutoff_len']:
                    valid, reason = False, 'student_overlong'
                counts[r['slot_id']] += 1
                record = dict(r, response=response, finish_reason=sample.finish_reason,
                              response_tokens=len(sample.token_ids), prompt_tokens=len(generation.prompt_token_ids),
                              attempt=counts[r['slot_id']], accepted=valid, rejection_reason=None if valid else reason,
                              student_total_tokens=student_length,
                              formatted_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                              generated_unix=time.time(), teacher=job['teacher'])
                raw.write(json.dumps(record, ensure_ascii=False) + '\n')
                if valid:
                    good.write(json.dumps(record, ensure_ascii=False) + '\n')
                    done.add(r['slot_id'])
                elif counts[r['slot_id']] < cfg['max_attempts']:
                    retry.append(r)
            raw.flush(); good.flush()
            pending = retry + pending
            write(out / f'worker{rank}_progress.json', {
                'accepted': len(done), 'target': quota, 'attempts': sum(counts.values()),
                'pending': len(pending), 'updated_unix': time.time(), 'quantity_mode': job['quantity_mode']})
    if job['quantity_mode'] == 'accepted_rows':
        assert len(done) == quota, 'Candidate pool exhausted before the contracted accepted count'
    write(out / f'worker{rank}_complete.json', {
        'accepted': len(done), 'attempts': sum(counts.values()),
        'accepted_sha256': digest(accepted_path), 'attempts_sha256': digest(attempts_path),
        'completed_unix': time.time()})


if __name__ == '__main__':
    main()

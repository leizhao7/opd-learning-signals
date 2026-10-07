"""Run the pinned LLaMA-Factory implementation and save extra audit observations."""
import argparse
import json
import math
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp

    class AuditCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            model = kwargs['model']
            d = {'started_unix': time.time(), 'pid': os.getpid(), 'world_size': args.world_size,
                 'global_batch': args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps,
                 'max_steps': state.max_steps, 'planned_epochs': args.num_train_epochs,
                 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            (Path(args.output_dir) / 'actual_start.json').write_text(json.dumps(d, indent=2))
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero:
                return
            for k in ['loss','grad_norm','eval_loss']:
                if k in (logs or {}):
                    assert math.isfinite(logs[k]), f'Non-finite {k}'
            with (Path(args.output_dir) / 'audit_metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': state.global_step, 'epoch': state.epoch,
                                    'time': time.time(), **(logs or {})}) + '\n')

    run_exp(config, callbacks=[AuditCallback()])
    if int(os.environ.get('RANK', '0')) == 0:
        out = Path(config['output_dir'])
        state = json.loads((out / 'trainer_state.json').read_text())
        assert state['global_step'] == state['max_steps']
        assert abs(state['epoch'] - config['num_train_epochs']) < 0.01
        assert (out / 'model.safetensors').exists() or (out / 'model.safetensors.index.json').exists()
        (out / 'training_complete.json').write_text(json.dumps({
            'completed_unix': time.time(), 'global_step': state['global_step'],
            'epoch': state['epoch'], 'full_final_audit_pending': True}, indent=2))


if __name__ == '__main__':
    main()

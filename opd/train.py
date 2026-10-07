"""Render an OPD recipe; execute only with --execute in a prepared GPU environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4

from .config import load_recipe, merge_overrides, recipe_names, render_overrides
from .paths import ROOT


def build_environment(vendor, output):
    env = os.environ.copy()
    paths = [ROOT / 'training/runtime', vendor, ROOT / 'evaluation/if']
    env['PYTHONPATH'] = os.pathsep.join([*(str(p) for p in paths), *filter(None, [env.get('PYTHONPATH')])])
    for key, value in {'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'true',
                       'HYDRA_FULL_ERROR': '1', 'PYTHONUNBUFFERED': '1'}.items():
        env.setdefault(key, value)
    env['VERL_FILE_LOGGER_PATH'] = str(output / 'metrics.jsonl')
    env['OUTLINES_CACHE_DIR'] = str(output / 'cache/outlines')
    env['MPLCONFIGDIR'] = str(output / 'cache/matplotlib')
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, choices=recipe_names())
    parser.add_argument('--student', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--val-data', required=True, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', default='opd-run')
    parser.add_argument('--override', action='append', default=[], help='Hydra key=value assignment; repeat as needed')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    vendor = ROOT / 'vendor' / recipe['vendor']
    output = args.output.resolve()
    # JSON quoting keeps commas, spaces and '=' inside user-provided scalar values.
    def quote(value):
        return json.dumps(value, ensure_ascii=False)
    values = {'REPO': str(ROOT), 'VENDOR': str(vendor), 'OUTPUT': str(output),
              'STUDENT_MODEL': args.student, 'TEACHER_MODEL': args.teacher,
              'TRAIN_DATA': str(Path(args.train_data).resolve()),
              'VAL_DATA': json.dumps([str(Path(p).resolve()) for p in args.val_data]),
              'RUN_NAME': args.name}
    try:
        overrides = render_overrides(recipe['overrides'], values)
        # Quote only substituted scalars, leaving lists and user Hydra expressions intact.
        overrides = [item.split('=', 1)[0] + '=' + quote(item.split('=', 1)[1])
                     if '${' in template and '${VAL_DATA}' not in template else item
                     for template, item in zip(recipe['overrides'], overrides)]
        overrides = merge_overrides(overrides, args.override)
    except ValueError as exc:
        parser.error(str(exc))
    command = [sys.executable, str(ROOT / 'training/runtime/train_entry.py'), *overrides]
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    for path in [args.train_data, *args.val_data]:
        if not Path(path).is_file():
            parser.error(f'Data file does not exist: {path}')
    output.mkdir(parents=True, exist_ok=True)
    env = build_environment(vendor, output)
    for key in ('OUTLINES_CACHE_DIR', 'MPLCONFIGDIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    record = {'recipe': recipe['name'], 'command': command, 'vendor': recipe['vendor'],
              'resolved_overrides': overrides, 'recipe_metadata': {k: v for k, v in recipe.items() if k != 'overrides'},
              'config_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT / 'configs/base.json', ROOT / f'configs/recipes/{args.recipe}.json')}}
    history = output / 'launches'
    history.mkdir(exist_ok=True)
    serialized = json.dumps(record, indent=2) + '\n'
    (history / f'{uuid4().hex}.json').write_text(serialized)
    (output / 'launch.json').write_text(serialized)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()

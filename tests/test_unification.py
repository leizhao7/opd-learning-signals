"""Protect historical settings and frozen backend behavior during organization."""
import gzip
import hashlib
import io
import tempfile
import json
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opd.config import load_recipe, merge_overrides, override_key, render_overrides
from opd.train import build_environment


class UnificationTests(unittest.TestCase):
    def test_all_settings_equal_to_pre_refactor_recipes(self):
        with gzip.open(ROOT / 'tests/fixtures/recipes_before_unification.json.gz', 'rt') as stream:
            originals = json.load(stream)
        for name, original in originals.items():
            with self.subTest(recipe=name):
                recipe = load_recipe(name)
                settings = lambda items: {override_key(item): item for item in items}
                self.assertEqual(settings(original['overrides']), settings(recipe['overrides']))
                self.assertEqual(original['vendor'], recipe['vendor'])
                self.assertEqual(original['source_sha256'], recipe['source_sha256'])

    def test_frozen_backends_are_byte_identical(self):
        hashes = json.loads((ROOT / 'provenance/frozen_vendor_sha256.json').read_text())
        actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (ROOT / 'vendor').rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        self.assertEqual(hashes, actual)

    def test_override_precedence(self):
        self.assertEqual(merge_overrides(['+a=1', 'b=2'], ['a=3', '+a=4']), ['+a=4', 'b=2'])
        with self.assertRaises(ValueError):
            merge_overrides([], ['invalid'])
        with self.assertRaises(ValueError):
            render_overrides(['a=${MISSING}'], {})
        with self.assertRaises(ValueError):
            load_recipe('../base')

    def test_environment_preserves_existing_search_paths(self):
        with patch.dict('os.environ', {'PYTHONPATH': '/custom', 'WANDB_MODE': 'offline'}, clear=True):
            env = build_environment(ROOT / 'vendor/verl-code', Path('/tmp/opd'))
        self.assertTrue(env['PYTHONPATH'].endswith('/custom'))
        self.assertEqual(env['WANDB_MODE'], 'offline')
        self.assertEqual(env['VERL_FILE_LOGGER_PATH'], '/tmp/opd/metrics.jsonl')

    def test_executed_launch_records_survive_resume(self):
        from opd.train import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data.parquet'
            data.touch()
            output = root / 'run'
            options = ['--recipe', 'code_rl', '--student', 'student', '--teacher', 'teacher',
                       '--train-data', str(data), '--val-data', str(data),
                       '--output', str(output), '--execute']
            with patch('opd.train.subprocess.run') as launch, patch('sys.stdout', new_callable=io.StringIO):
                main(options)
                main([*options, '--override', 'trainer.total_training_steps=300'])
            self.assertEqual(launch.call_count, 2)
            records = [json.loads(p.read_text()) for p in (output / 'launches').glob('*.json')]
            self.assertEqual(len(records), 2)
            latest = json.loads((output / 'launch.json').read_text())
            self.assertIn('trainer.total_training_steps=300', latest['resolved_overrides'])
            self.assertTrue(any('trainer.total_training_steps=200' in r['resolved_overrides'] for r in records))
            self.assertEqual(len(latest['config_sha256']), 2)
            self.assertTrue((output / 'cache/outlines').is_dir())

    def test_unified_and_legacy_cli_match_and_hydra_parses_paths(self):
        try:
            from hydra.core.override_parser.overrides_parser import OverridesParser
        except ImportError:
            self.skipTest('Install requirements-test.txt to check Hydra parsing')
        options = ['--recipe', 'code_rl', '--student', '/models/a,b = c', '--teacher', '/models/teacher',
                   '--train-data', '/data/a,b = c.parquet', '--val-data', '/data/one file.parquet',
                   '/data/two.parquet', '--output', '/tmp/opd output', '--name', 'run, a=b',
                   '--override', 'trainer.total_training_steps=3']
        def command(entry):
            result = subprocess.run([sys.executable, *entry, *options], cwd=ROOT, capture_output=True, text=True, check=True)
            return shlex.split(result.stdout.strip())
        unified = command(['-m', 'opd', 'train'])
        self.assertEqual(unified, command([str(ROOT / 'scripts/train.py')]))
        parsed = OverridesParser.create().parse_overrides(unified[2:])
        values = {item.key_or_group: item.value() for item in parsed}
        self.assertEqual(values['actor_rollout_ref.model.path'], '/models/a,b = c')
        self.assertEqual(values['data.train_files'], '/data/a,b = c.parquet')
        self.assertEqual(values['trainer.experiment_name'], 'run, a=b')
        self.assertEqual(values['data.val_files'], ['/data/one file.parquet', '/data/two.parquet'])
        self.assertEqual(values['trainer.total_training_steps'], 3)


if __name__ == '__main__':
    unittest.main()

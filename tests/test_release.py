import importlib.util,json,re,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opd.backends import backend_path
from opd.config import load_recipe
spec=importlib.util.spec_from_file_location('signals',ROOT/'analysis/learning_signal.py');signals=importlib.util.module_from_spec(spec);spec.loader.exec_module(signals)

class ReleaseTests(unittest.TestCase):
    def test_factorization(self):
        for loss,g2,cross in [(2,3,1),(2,3,-4),(2,-1,.5)]:
            s=signals.crossfit_statistics(loss,g2,cross)
            self.assertAlmostEqual(s['gamma'],s['mu']*(1+s['alpha']))
        self.assertFalse(signals.crossfit_statistics(2,-1,.5)['positive_g2'])
    def test_invalid_loss(self):
        with self.assertRaises(ValueError):signals.crossfit_statistics(0,1,1)
    def test_all_recipes_dry_run(self):
        for p in (ROOT/'configs/recipes').glob('*.json'):
            recipe=load_recipe(p.stem);vendor=backend_path(recipe['vendor'])
            self.assertTrue((vendor/'verl/version/version').is_file())
            r=subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--recipe',p.stem,'--student','student','--teacher','teacher','--train-data','/data/train.parquet','--val-data','/data/a.parquet','/data/b.parquet','--output','/tmp/opd-release-dry-run','--override','trainer.total_training_steps=3'],text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertNotIn('${',r.stdout)
            self.assertNotIn('/workspace/lei',r.stdout)
            self.assertIn('trainer.total_training_steps=3',r.stdout)
            self.assertNotIn('trainer.total_training_steps=200',r.stdout)
    def test_no_internal_paths_or_credentials(self):
        # Scans every shipped text file, including vendored and provenance sources.
        pattern=re.compile('|'.join([r'/workspace/',r'migrated_from_\d',r'XSearcher',r'old_home',r'\bZQ\b',r'\bPVC\b',
            r'hf_[A-Za-z0-9]{30,}',r'sk-[A-Za-z0-9]{32,}',r'AKIA[0-9A-Z]{16}',r'ghp_[A-Za-z0-9]{36}']))
        skip={'.git','.build','__pycache__','.mplcache','.venv'}
        found=[]
        for p in ROOT.rglob('*'):
            if not p.is_file() or skip & set(p.relative_to(ROOT).parts) or p==Path(__file__).resolve():continue
            try:text=p.read_text(encoding='utf-8')
            except UnicodeDecodeError:continue
            found+=[f'{p.relative_to(ROOT)}: {m.group(0)}' for m in pattern.finditer(text)]
        self.assertEqual(found,[])
    def test_recipe_input_paths(self):
        for p in (ROOT/'configs/recipes').glob('*.json'):
            d=load_recipe(p.stem)
            for item in d['overrides']:
                if item.startswith('custom_reward_function.path='):
                    v=item.split('=',1)[1].replace('${REPO}',str(ROOT)).replace('${VENDOR}',str(backend_path(d['vendor'])))
                    self.assertTrue(v=='null' or Path(v).is_file(),v)
if __name__=='__main__':unittest.main()

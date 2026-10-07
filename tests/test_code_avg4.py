import collections,csv,gzip,hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class SamplingAudit(unittest.TestCase):
    def test_archived_selection_and_all_21_accuracies(self):
        grouped=collections.defaultdict(lambda:collections.defaultdict(list))
        identities={}
        with gzip.open(ROOT/'paper/figures/data/code_avg4_selected.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line)
                ranks=[(hashlib.sha256(('seed=0\0'+r['benchmark']+'\0'+r['prompt_sha256']+'\0'+str(i)).encode()).hexdigest(),i) for i in range(8)]
                indices=sorted(i for _,i in sorted(ranks)[:4])
                self.assertEqual(indices,r['selected_response_indices_zero_based'])
                self.assertEqual([r['rewards'][i] for i in indices],r['selected_rewards'])
                key=(r['benchmark'],r['prompt_sha256'])
                if key in identities:self.assertEqual(indices,identities[key])
                identities[key]=indices
                grouped[r['step']][r['benchmark']].append(sum(r['selected_rewards'])/4)
        self.assertEqual(sorted(grouped),list(range(0,201,10)))
        with (ROOT/'paper/figures/data/opd_teacher_trainability_multigroup.csv').open() as f:
            expected={int(r['step']):float(r['macro_val_accuracy_percent']) for r in csv.DictReader(f) if r['teacher']=='rl4b' and r['macro_val_accuracy_percent']}
        for step,benches in grouped.items():
            score=sum(sum(x)/len(x) for x in benches.values())/len(benches)*100
            self.assertAlmostEqual(score,expected[step],places=10)
if __name__=='__main__':unittest.main()

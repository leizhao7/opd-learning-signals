"""Pin the existing DAPO questions; replace only the old answer-format wrapper."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True,help='Working directory containing contract.proposed.json')
parser.add_argument('--source',type=Path,required=True,help='The exact original DAPO parquet')
args=parser.parse_args()
ROOT=args.root.resolve();ROOT.mkdir(parents=True,exist_ok=True)
source=args.source.resolve()
expected='039f3afd689c846985bd2bf58e55a2210a8b08a1a9e1f60dbc07107a5f341925'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from llamafactory.hparams import DataArguments
from llamafactory.data import get_template_and_fix_tokenizer
import torch

data=ROOT/'data';data.mkdir(exist_ok=True)
assert not (data/'manifest.json').exists(),'Prompt manifest already finalized; inspect before changing'
shutil.copyfile(source,data/'dapo-math-17k.source.parquet')
rows=pq.read_table(source).to_pylist();assert len(rows)==17917
prefix='Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
suffix='\n\nRemember to put your answer on its own line after "Answer:".'
instruction=r'Please reason step by step, and put your final answer within \boxed{}.'
questions=[]
for i,row in enumerate(rows):
 messages=row['prompt'];assert len(messages)==1 and messages[0]['role']=='user'
 text=messages[0]['content'];assert text.startswith(prefix) and text.endswith(suffix),i
 body=text[len(prefix):-len(suffix)]
 assert prefix+body+suffix==text and body.strip()
 questions.append({'source_row_id':i,'source_id':row['extra_info']['index'],
   'question_sha256':hashlib.sha256(body.encode()).hexdigest(),
   'original_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
   'ground_truth':row['reward_model']['ground_truth'],
   'messages':[{'role':'user','content':body+'\n\n'+instruction}]})
assert len({q['source_id'] for q in questions})==len(questions)
contract=json.loads((ROOT/'contract.proposed.json').read_text())
lengths={}
for job in [contract['jobs'][0],contract['jobs'][2]]:
 tok=AutoTokenizer.from_pretrained(job['student'],local_files_only=True,trust_remote_code=False)
 template=get_template_and_fix_tokenizer(tok,DataArguments(template=job['sft_config']['template'],enable_thinking=True))
 vals=[]
 for row in questions:
  x,_=template.encode_oneturn(tok,row['messages']+[{'role':'assistant','content':r'\boxed{0}'}])
  vals.append(len(x))
 lengths[job['model_family']]={'max':max(vals),'over2048':sum(v>2048 for v in vals)}
 # The rollout checks the actual student target length before accepting every row.
 assert max(vals)<32768-12288,'Some question exceeds the teacher context budget'
(data/'questions.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in questions))
slot=0
with (data/'prompts.jsonl').open('w') as f:
 for cycle in range(4):
  order=list(range(len(questions)));random.Random(42+cycle).shuffle(order)
  for idx in order:
   f.write(json.dumps(dict(questions[idx],slot_id=slot,cycle=cycle),ensure_ascii=False)+'\n');slot+=1
manifest={'dataset':'existing project DAPO math train','source_file':str(source),'source_sha256':expected,
 'source_rows':len(rows),'unique_source_ids':len(questions),'unique_question_texts':len({q['question_sha256'] for q in questions}),
 'prompt_sha256':hashlib.sha256((data/'prompts.jsonl').read_bytes()).hexdigest(),
 'questions_sha256':hashlib.sha256((data/'questions.jsonl').read_bytes()).hexdigest(),
 'candidate_slots':slot,'candidate_cycles':4,'shuffle_seed_per_cycle':'42 + cycle',
 'target_accepted_rows_per_teacher':20000,'num_rollouts_per_attempt':1,'max_attempts_per_slot':3,
 'question_body_preserved':True,'removed_prefix':prefix,'removed_suffix':suffix,'appended_instruction':instruction,
 'correctness_filter':False,'prompt_lengths':lengths,'cuda_initialized':torch.cuda.is_initialized(),'time':time.time()}
assert not manifest['cuda_initialized']
(data/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False),flush=True)

import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
for d in ['data','results','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
sources={'company':'jspace_relation_binding','four':'jspace_cross_nohint','five':'jspace_five_scenarios'}
for key,folder in sources.items():
 source=BASE/folder;cfg=read(source/'protocol.json');jobs=read(source/'data/jobs.json');byid={j['job_id']:j for j in jobs}
 paths=[source/'results'/f'{s}.jsonl' for s in ['development','validation']] if key=='company' else [source/'results/outputs.jsonl']
 old=[json.loads(line) for p in paths for line in p.read_text(encoding='utf-8').splitlines()]
 selected=[byid[r['job_id']] for r in old];assert len(selected)=={'company':924,'four':1152,'five':1440}[key]
 save(f'data/{key}_jobs.json',selected);save(f'data/{key}_config.json',cfg)
 shutil.copyfile(source/'scoring.py',ROOT/f'{key}_scoring.py');shutil.copyfile(source/'analyze.py',ROOT/f'{key}_analyze.py')
 with (ROOT/f'data/8b_{key}.jsonl').open('w',encoding='utf-8') as f:
  for r in old:f.write(json.dumps(r,ensure_ascii=False)+'\n')
shutil.copyfile(BASE/'jspace_five_small/results/answer_outputs.jsonl',ROOT/'data/2b_five.jsonl')
shutil.copyfile(BASE/'jspace_five_small/results/transport_audit.json',ROOT/'data/2b_five_transport_audit.json')
save('protocol.json',dict(models={'2b':'models/Qwen3.5-2B','27b':'models/Qwen3.5-27B'},datasets=sources,
 schedule=[dict(model='2b',datasets=['company','four'],n=2076),dict(model='27b',datasets=['company','four','five'],n=3516)],
 reused='Qwen3.5-2B five scenarios: 1440 existing generations, native turn boundary normalized and audited. Qwen3-8B source runs retained for comparison.',
 scope='Ten relation scenarios. Company: latest relation-binding formal dataset, all924 runs, no70pilot. Four: no-hint version. Five: locked new scenario corpus. Prior repeated difficulty variants and causal patches are not distinct scenarios and are not rerun.',
 settings='BF16 eager, native non-thinking chat, greedy, max128 new tokens; same source jobs and generation order. Company batch12 and other datasets batch16. Model generation ends on either tokenizer EOS or native im_end.',
 controls='No prompt edits, no additional relationship warnings, no revised scoring or screening thresholds. Original company dev/validation retained for reporting, already observed historically.',
 limits='Same-generation 2B/27B contrast still changes architecture/training, not a causal size-only experiment. Company design differs from other nine; do not pool their rates.',
 python='./llm_knowledge/.venv/bin/python'))
save('data/lock.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'data').glob('*'))+list(ROOT.glob('*scoring.py'))+list(ROOT.glob('*analyze.py'))+[ROOT/'run.py',ROOT/'protocol.json'] if p.name!='lock.json'})
print('Locked 2076 small-model additions then 3516 large-model runs.')

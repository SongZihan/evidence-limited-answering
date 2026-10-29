import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'jspace_cross_scenario'
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads((OLD/p).read_text(encoding='utf-8'))
for d in ['data','results','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
hints={'person':'工作、就学与出生关系不能互相替代。','event':'机构地址、往届地点不能替代本届举办地点。','product':'销售、仓储与生产关系不能互相替代。','research':'任职、提供数据与提供经费资助是不同关系，不能互相替代。'}
jobs=read('data/jobs.json')
for j in jobs:
 before=dict(j);h=hints[j['family']];assert j['user_prompt'].count(h)==1
 j['user_prompt']=j['user_prompt'].replace(h,'')
 assert all(j[k]==v for k,v in before.items() if k!='user_prompt')
save('data/jobs.json',jobs)
for name in ['run.py','scoring.py','analyze.py']:
 shutil.copyfile(OLD/name,ROOT/name)
for name in ['materials.json','materials.csv','prior_environment.json']:shutil.copyfile(OLD/'data'/name,ROOT/'data'/name)
shutil.copyfile(OLD/'results/outputs.jsonl',ROOT/'data/with_hint_outputs.jsonl')
shutil.copyfile(OLD/'results/environment.json',ROOT/'data/with_hint_environment.json')
cfg=read('protocol.json')
for family,h in hints.items():
 assert cfg['families'][family]['definitions'].count(h)==1
 cfg['families'][family]['definitions']=cfg['families'][family]['definitions'].replace(h,'')
cfg['paired_rerun']=dict(only_prompt_change=hints,unchanged='Materials, gold, job order, seed, batches, model, decoding, scoring and discovery thresholds unchanged.',comparison='Paired no-hint minus with-hint source misuse; retain original outcomes. Exploratory sensitivity analysis, not fresh validation.')
save('protocol.json',cfg)
save('data/prompt_change_audit.json',dict(n=1152,only_user_prompt_changed=True,deleted_sentences=hints,old_jobs_sha256=hashlib.sha256((OLD/'data/jobs.json').read_bytes()).hexdigest(),old_outputs_sha256=hashlib.sha256((OLD/'results/outputs.jsonl').read_bytes()).hexdigest()))
save('data/lock.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['protocol.json','data/jobs.json','data/materials.json','run.py','scoring.py','analyze.py','data/with_hint_outputs.jsonl','data/prompt_change_audit.json']})
print('Locked 1152 paired jobs, only domain-specific hint removed.')

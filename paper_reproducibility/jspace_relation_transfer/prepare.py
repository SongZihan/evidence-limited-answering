import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for d in ['data','results','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
pairs=[];coverage=[]
for ds in ['company','four','five','round1']:
 src=ROOT.parent/('jspace_relation_mechanism_round1/data/jobs.json' if ds=='round1' else f'jspace_more_models/data/{ds}_jobs.json')
 jobs=read(src);groups={}
 for j in jobs:
  if j['condition'] not in ['B','D']:continue
  x=dict(j);x['dataset']=ds;x['uid']=ds+':'+j['job_id'];x['family']=j.get('family','company')
  x['form']=j.get('relation_form',j.get('form'));x['evidence']=j.get('evidence_version',j.get('evidence','main'))
  if x['evidence']=='absent':x['evidence']='main'
  x['target_field']=j.get('target_field','总部城市');x['target_order']=j.get('target_order','first');x['split']=j.get('split','historical_exploratory')
  x['source_value']=j.get('source_value',j.get('office_city'));x['control_value']=j.get('control_value',j.get('launch_city',j.get('city_b')))
  key=(j['group_id'],x['form'],x['evidence'],x['target_order']);groups.setdefault(key,{})[j['condition']]=x
 for key,g in sorted(groups.items()):
  assert set(g)=={'B','D'},(ds,key)
  assert g['B']['material']==g['D']['material'] and g['B']['gold'][g['B']['target_field']]==g['D']['gold'][g['D']['target_field']]
  pairs.append(dict(pair_id=ds+':'+ '|'.join(key),B=g['B'],D=g['D']))
 covered={p['B']['material'] for p in pairs if p['B']['dataset']==ds}
 assert set(j['material'] for j in jobs)<=covered
 coverage.append(dict(dataset=ds,source_records=len(jobs),paired_materials=len(covered),pairs=len(groups),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),uncovered_materials=0))
assert len(pairs)==1176
for i,p in enumerate(pairs):p['worker']=i%2
save('data/pairs.json',pairs);save('data/coverage.json',coverage)
prior=ROOT.parent/'jspace_relation_mechanism_round1'
for src,dst in [('results/frozen_basis.pt','data/frozen_basis.pt'),('results/random_basis.pt','data/random_basis.pt'),('results/selection.json','data/original_selection.json'),('data/prior_environment.json','data/prior_environment.json')]:shutil.copyfile(prior/src,ROOT/dst)
save('protocol.json',dict(model='models/Qwen3-8B',layer=33,rank=4,seed=20261015,batch_size=1,max_new_tokens=128,workers=2,pairs=1176,
 scope='984 historical pairs across ten scenes plus192 round1 pairs. All source materials covered; A/C prompts not separate intervention conditions. Company first/last and diagnostic forms included.',
 frozen='Reuse round1 basis and seeded random basis without refitting, changing rank/layer, or screening outcomes.',
 procedure='Natural B/D, exact-natural-prefix replay, selfB/selfD, same-material projected B->D and D->B, fullB->D, norm-matched randomB->D. Target-relative PN, recipient cache. One sample per inference, independent paired jobs split evenly across two GPUs.',
 scoring='Strict single JSON object, ordered required fields, scalar target string/null (numeric gold also allows number); strings null count as errors. Source misuse and other wrong values separate. Supported source=gold not counted as misuse. No fencing repair.',
 coverage='If either natural PN unavailable, retain both natural rows, flag pair excluded from intervention; no replacement of missing positions. Report outcome and protocol failures.',
 interpretation='Historical material and round1 development are not new independent validation. Round1 validation has now been observed. No new confirmatory success gate; descriptive transfer, repair/harm and per-template counts, no large statistical scans.',
 numerical='Single-sample BF16 eager greedy. Replay serves as causal baseline. Random perturbation norm matched before BF16 cast; record intended and realized norms. No automatic retry on changed prompts or token budgets.'))
save('data/lock.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'protocol.json',*sorted((ROOT/'data').glob('*'))] if p.name!='lock.json'})
print(json.dumps(coverage,ensure_ascii=False));print('pairs',len(pairs))

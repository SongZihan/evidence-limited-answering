from pathlib import Path
import json,collections,hashlib
root=Path(__file__).resolve().parents[2];out=[]
def read(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def js(p):return json.loads(p.read_text(encoding='utf-8'))
spec=[('Qwen3.5-2B','jspace_multimodel_intervention/results/qwen2b'),('Qwen3.5-27B','jspace_multimodel_intervention/results/qwen27b'),('Gemma-4-12B','jspace_family_intervention/results/gemma'),('Muse-Glimmer-30B','jspace_family_intervention/results/muse'),('Kimi-VL-A3B','jspace_family_intervention/results/kimi'),('GLM-4.1V-9B','jspace_family_intervention/results/glm')]
for name,path in spec:
 p=root/path;val=js(p/'validation_summary.json');rs=read(p/'pairs.jsonl');checks=0
 for pair in rs:
  for r in pair['rows']:
   if r['kind'] in ['self_B','self_D']:
    ref=next(z for z in pair['rows'] if z['kind']=='replay' and z['uid']==r['uid']);assert r['generated_token_ids']==ref['generated_token_ids'];checks+=1
 subset=[r for r in rs if r['dataset']=='round1' and r['split']=='validation' and r['form']=='weak'];assert len(subset)==48
 controls={}; transitions={}
 for ev in ['main','supported','swapped']:
  pairs=[r for r in subset if r['evidence']==ev and r['available']];sm=val['evidence'][ev];assert len(pairs)==sm['available']
  counts={}
  for kind in ['replay','forward','random','full','reverse']:
   rr=[next(z for z in r['rows'] if z['kind']==kind and (kind!='replay' or z['uid'].endswith('_D'))) for r in pairs];counts[kind]=sum(z['correct'] for z in rr)
  assert counts['replay']==sm['before_correct'] and counts['forward']==sm['after_correct']
  controls[ev]=counts
 out.append(dict(model=name,source=path,validation=val,controls=controls,selection=js(p/'selection.json')['selected'],self_checks=checks,audit=js(p/'audit.json'),history=js(p/'main_summary.json'),sha256=hashlib.sha256((p/'pairs.jsonl').read_bytes()).hexdigest()))
p=root/'jspace_relation_mechanism_round1/results';jobs={j['job_id']:j for j in js(root/'jspace_relation_mechanism_round1/data/jobs.json')};ints=read(p/'interventions.jsonl');replay={r['job_id']:r for r in read(p/'replay.jsonl')};controls={};evidence={};checks=0
for r in ints:
 if r['kind']=='self':assert r['generated_token_ids']==replay[r['job_id']]['generated_token_ids'];checks+=1
for ev in ['absent','supported','swapped']:
 rr=[r for r in ints if r['split']=='validation' and r['evidence']==ev and jobs[r['job_id']]['form']=='weak'];forward=[r for r in rr if r['kind']=='forward'];assert len(forward)==16
 counts={k:sum(r['correct'] for r in rr if r['kind']==k) for k in ['forward','random','full','reverse']};counts['replay']=sum(replay[r['job_id']]['correct'] for r in forward)
 evidence['main' if ev=='absent' else ev]={'available':16,'planned':16,'before_correct':counts['replay'],'after_correct':counts['forward']};controls['main' if ev=='absent' else ev]=counts
out.insert(1,dict(model='Qwen3-8B',source=str(p.relative_to(root)),validation={'passed':False,'evidence':evidence},controls=controls,selection={'layer':33,'rank':4},self_checks=checks,history=js(root/'jspace_relation_transfer/results/main_summary.json')))
dest=root/'formal/_draft_work/section44_data.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for m in out:print(m['model'],m['selection']['layer'],m['selection']['rank'],m['validation']['passed'],m['controls'],'self',m['self_checks'])

from pathlib import Path
import json,statistics as st,hashlib
root=Path(__file__).resolve().parents[2];base=root/'jspace_relation_mechanism_round1';dest=root/'formal/_draft_work'
def read(n):return [json.loads(l) for l in (base/'results'/n).read_text(encoding='utf-8').splitlines() if l.strip()]
jobs={j['job_id']:j for j in json.loads((base/'data/jobs.json').read_text(encoding='utf-8'))};valid={k:j for k,j in jobs.items() if j['split']=='validation'};assert len(valid)==192
rr={(r['job_id'],r['position'],r['method'],r['layer']):r for r in read('readouts.jsonl')}
env=json.loads((base/'results/environment.json').read_text(encoding='utf-8'));assert all(len(env['tokens'][c])==1 for c in ['武汉','成都','上海','南京'])
coverage=[r for r in read('coverage.jsonl') if r['job_id'] in valid];assert len(coverage)==192 and all(r['available'] for r in coverage)
layers=[9,13,17,21,25,29,33];curves=[];swaps=[];cs=read('candidates.jsonl');cand=[]
for method in ['J-lens','Logit Lens']:
 for pos in ['P0','PN']:
  for layer in layers:
   vals=[]
   for j in valid.values():
    if (j['evidence'],j['form'],j['condition'])!=('absent','weak','B'):continue
    k=j['job_id'];kd=k[:-1]+'D';assert valid[kd]['material']==j['material']
    def v(k):r=rr[(k,pos,method,layer)]['scores'];return r[j['city_a']]-r[j['city_b']]
    vals.append(v(kd)-v(k))
   assert len(vals)==16;curves.append(dict(method=method,position=pos,layer=layer,mean=st.mean(vals),values=vals))
 for cond in ['B','D']:
  pairs=[]
  for j in valid.values():
   if (j['evidence'],j['form'],j['condition'])!=('supported','weak',cond):continue
   ks=j['job_id'].replace('_supported_','_swapped_');sj=valid[ks];assert (j['city_a'],j['city_b'])==(sj['city_a'],sj['city_b'])
   def v(k):s=rr[(k,'PN',method,33)]['scores'];return s[j['city_a']]-s[j['city_b']]
   pairs.append(dict(group_id=j['group_id'],before=v(j['job_id']),after=v(ks)))
  swaps.append(dict(method=method,condition=cond,n=len(pairs),both_correct=sum(p['before']<0 and p['after']>0 for p in pairs),before_mean=st.mean(p['before'] for p in pairs),after_mean=st.mean(p['after'] for p in pairs),pairs=pairs))
for evidence in ['absent','supported','swapped']:
 for form in ['weak','direct']:
  for cond in ['B','D']:
   vals=[];ranks=[]
   for j in valid.values():
    if (j['evidence'],j['form'],j['condition'])!=(evidence,form,cond):continue
    c=[r for r in cs if r['job_id']==j['job_id']];assert len(c)==3 and all(r['position']=='P1' for r in c)
    assert abs(sum(r['candidate_set_probability'] for r in c)-1)<1e-5
    good=next(r for r in c if r['value']==j['hq_city']);vals.append(good['candidate_set_probability']);ranks.append(max(c,key=lambda r:r['logp'])['value']==j['hq_city'])
   assert len(vals)==16;cand.append(dict(evidence=evidence,form=form,condition=cond,mean=st.mean(vals),values=vals,top=sum(ranks)))
natural=[r for r in read('natural.jsonl') if r['job_id'] in valid];behavior=[]
for ev in ['absent','supported','swapped']:
 for form in ['weak','direct']:
  for c in ['B','D']:
   rows=[r for r in natural if (r['evidence'],r['form'],r['condition'])==(ev,form,c)];assert len(rows)==16
   behavior.append(dict(evidence=ev,form=form,condition=c,correct=sum(r['correct'] for r in rows),office=sum(r['office_misuse'] for r in rows)))
data=dict(scope='Qwen3-8B validation only; all 192 natural runs; no outcome selection',coverage=len(coverage),curves=curves,swaps=swaps,candidates=cand,behavior=behavior,tokens=env['tokens'],source_hashes={n:hashlib.sha256((base/'results'/n).read_bytes()).hexdigest() for n in ['readouts.jsonl','natural.jsonl','candidates.jsonl','coverage.jsonl']})
(dest/'section43_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('Layer33 gaps',[(r['method'],r['position'],round(r['mean'],3)) for r in curves if r['layer']==33]);print('Swaps',[(r['method'],r['condition'],r['both_correct'],r['before_mean'],r['after_mean']) for r in swaps]);print('Candidates',[(r['evidence'],r['form'],r['condition'],round(r['mean']*100,2),r['top']) for r in cand])

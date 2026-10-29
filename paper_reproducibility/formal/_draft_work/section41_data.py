from pathlib import Path
import json,random,statistics,csv,hashlib
root=Path(__file__).resolve().parents[2]
spec=[('Qwen2.5-1.5B','jspace_small_models_scope/results/qwen15_company.jsonl',None),('Gemma-2-2B','jspace_small_models_scope/results/gemma2_company.jsonl',None),('Qwen3.5-2B','jspace_all_models/results/2b_company.jsonl',None),('Qwen2.5-3B','jspace_small_models_scope/results/qwen3_company.jsonl',None),('Llama-3.2-3B','jspace_small_models_scope/results/llama3_company.jsonl',None),('Phi-4-mini','jspace_small_models_scope/results/phi4mini_company.jsonl',None),('Qwen3-4B','jspace_small_models_scope/results/qwen4_company.jsonl',None),('Qwen3-8B','jspace_relation_binding/results/validation.jsonl',None),('GLM-4.1V-9B','jspace_more_models/results/glm9b_company.jsonl','jspace_more_models/results/glm9b_company_content.jsonl'),('Gemma-4-12B','jspace_gemma12b/results/gemma12b_company.jsonl','jspace_gemma12b/results/gemma12b_company_fence_view.jsonl'),('Kimi-VL-A3B','jspace_more_models/results/kimi16b_company_native.jsonl','jspace_more_models/results/kimi16b_company_content.jsonl'),('Qwen3.5-27B','jspace_all_models/results/27b_company.jsonl',None),('Muse-Glimmer-30B','jspace_more_models/results/muse30b_company.jsonl','jspace_more_models/results/muse30b_company_content.jsonl')]
def read(p):return [json.loads(l) for l in (root/p).read_text(encoding='utf-8').splitlines() if l.strip()]
def summary(rs):
 rs=[r for r in rs if r['split']=='validation'];cells={}; ts=sorted({r['template_id'] for r in rs})
 assert len(rs)==462 and len(ts)==6
 for form in ['weak','direct']:
  for c in 'ABCD':
   x=[r for r in rs if r['relation_form']==form and r['condition']==c]; assert len(x)==(18 if c=='A' else 36)
   cells[form+'_'+c]={'n':len(x),'source':sum(r.get('office_city_misuse') is True for r in x),'failure':sum(r.get('protocol_failure') is True for r in x),'null':sum(r.get('reliable_null') is True for r in x),'truncated':sum(r.get('truncated_generation') is True for r in x)}
 diffs={}
 for form in ['weak','direct']:
  diffs[form]=[]
  for t in ts:
   x=[r for r in rs if r['relation_form']==form and r['template_id']==t]
   vals={c:sum(r.get('office_city_misuse') is True for r in x if r['condition']==c)/6*100 for c in 'BD'}
   diffs[form].append(vals['D']-vals['B'])
 inter=[a-b for a,b in zip(diffs['weak'],diffs['direct'])];rng=random.Random(20261025)
 boot=sorted(statistics.mean(rng.choices(inter,k=6)) for _ in range(2000))
 x=[r for r in rs if r['relation_form']=='supported']
 return dict(cells=cells,templates=ts,template_diffs=diffs,weak_gap=statistics.mean(diffs['weak']),direct_gap=statistics.mean(diffs['direct']),interaction=statistics.mean(inter),interaction_ci=[boot[49],boot[1949]],positive_templates=sum(v>0 for v in diffs['weak']),supported_correct=sum(r.get('supported_correct') is True for r in x),supported_n=len(x))
out=[];reference=None
for name,path,aux in spec:
 rows=read(path);valid=[r for r in rows if r['split']=='validation'];mapping={r['job_id']:(r['material'],r['user_prompt']) for r in valid}
 if reference is None:reference=mapping
 assert mapping==reference,(name,'prompt mismatch')
 for form in ['weak','direct']:
  by={r['job_id']:r for r in valid if r['relation_form']==form}
  for r in by.values():
   if r['condition']=='B':
    donor=next(z for z in by.values() if z['condition']=='D' and z['group_id']==r['group_id'] and z['target_order']==r['target_order'])
    assert r['material']==donor['material']
 content=read(aux) if aux else [dict(r,**r.get('content_score',{})) for r in rows]
 out.append(dict(model=name,source=path,sha256=hashlib.sha256((root/path).read_bytes()).hexdigest(),strict=summary(rows),content=summary(content)))
dest=root/'formal/_draft_work/section41_data.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for m in out:
 s=m['strict'];c=m['content'];print(m['model'],'strict gaps',round(s['weak_gap'],1),round(s['direct_gap'],1),'interaction',round(s['interaction'],1),'CI',s['interaction_ci'],'positive',s['positive_templates'],'content gaps',round(c['weak_gap'],1),round(c['direct_gap'],1),'support',s['supported_correct'],c['supported_correct'])

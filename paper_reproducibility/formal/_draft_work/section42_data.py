from pathlib import Path
import json, hashlib
root=Path(__file__).resolve().parents[2]
def read(p): return [json.loads(l) for l in (root/p).read_text(encoding='utf-8').splitlines() if l.strip()]
prev=json.loads((root/'formal/_draft_work/section41_data.json').read_text(encoding='utf-8'))
out=[]; reference={}
for m in prev:
 name=m['model']; cp=m['source']; paths={}
 if name=='Qwen3-8B': paths={'four':'jspace_cross_nohint/results/outputs.jsonl','five':'jspace_five_scenarios/results/outputs.jsonl'}
 elif name=='Qwen3.5-2B': paths={'four':'jspace_all_models/results/2b_four.jsonl','five':'jspace_all_models/data/2b_five.jsonl'}
 else:
  for part in ['four','five']:paths[part]=cp.replace('company',part)
 cells={'company':m['strict']};sources=[]
 for part,path in paths.items():
  rs=read(path);sources.append({'path':path,'sha256':hashlib.sha256((root/path).read_bytes()).hexdigest()})
  mapping={r['job_id']:(r['material'],r['user_prompt']) for r in rs}
  assert len(mapping)==len(rs)
  if part not in reference:reference[part]=mapping
  assert mapping==reference[part],(name,part,'prompt mismatch')
  for family in sorted({r['family'] for r in rs}):
   fs=[r for r in rs if r['family']==family]; main=[r for r in fs if r['evidence']=='main'];ts=sorted({r['template_id'] for r in main}); assert len(ts)==8
   c={}
   for form in ['weak','direct']:
    for cond in 'ABCD':
     a=[r for r in main if r['form']==form and r['condition']==cond];assert len(a)==32
     c[form+'_'+cond]={'n':len(a),'source':sum(r['source_misuse'] for r in a),'failure':sum(not r['protocol_valid'] for r in a),'control':sum(r['control_misuse'] for r in a),'null':sum(r['true_null'] for r in a)}
   td={form:[100*(sum(r['source_misuse'] for r in main if r['form']==form and r['template_id']==t and r['condition']=='D')-sum(r['source_misuse'] for r in main if r['form']==form and r['template_id']==t and r['condition']=='B'))/4 for t in ts] for form in ['weak','direct']}
   gaps={form:sum(td[form])/8 for form in td}
   for b in [r for r in main if r['condition']=='B']:
    dr=next(r for r in main if r['condition']=='D' and r['group_id']==b['group_id'] and r['form']==b['form']);assert b['material']==dr['material']
   sup=[r for r in fs if r['evidence']=='supported']
   cells[family]={'cells':c,'templates':ts,'template_diffs':td,'weak_gap':gaps['weak'],'direct_gap':gaps['direct'],'interaction':gaps['weak']-gaps['direct'],'positive_templates':sum(v>0 for v in td['weak']),'supported_correct':sum(r['target_correct'] for r in sup),'supported_n':len(sup)}
 out.append({'model':name,'scenarios':cells,'sources':sources})
(root/'formal/_draft_work/section42_data.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for m in out:
 print(m['model'])
 for f,s in m['scenarios'].items():
  c=s['cells'];fail=sum(c['weak_'+x]['failure'] for x in 'ABCD')/sum(c['weak_'+x]['n'] for x in 'ABCD')*100
  print(f, 'gap',round(s['weak_gap'],1),'interaction',round(s['interaction'],1),'positive',s['positive_templates'],'failure',round(fail,1),'support',str(s['supported_correct'])+'/'+str(s['supported_n']))

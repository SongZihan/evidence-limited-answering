import statistics,random
from collections import Counter
def analyze(rows,cfg):
 rng=random.Random(cfg['seed']);weights=[[rng.randrange(8) for _ in range(8)] for _ in range(cfg['bootstrap']['replicates'])]
 def interval(mapping):
  values=[v for k,v in sorted(mapping.items())];assert len(values)==8
  avg=statistics.mean(values);samples=sorted(sum(values[i] for i in w)/8 for w in weights)
  return dict(mean=avg,ci95=[samples[49],samples[1949]],template_values=mapping,same_sign_templates=sum(v*avg>0 for v in values))
 result={};threshold=cfg['discovery_rule']
 for family in cfg['families']:
  rs=[r for r in rows if r['family']==family];main=[r for r in rs if r['evidence']=='main'];templates=sorted({r['template_id'] for r in main})
  cells=[]
  for form in ['weak','direct','missing','supported']:
   for cond in ('ABCD' if form in ['weak','direct'] else 'BD'):
    cell=[r for r in rs if r['form']==form and r['condition']==cond]
    cells.append(dict(form=form,condition=cond,n=len(cell),source=sum(r['source_misuse'] for r in cell),control=sum(r['control_misuse'] for r in cell),other=sum(r['other_string'] and r['evidence']!='supported' for r in cell),null=sum(r['true_null'] for r in cell),string_null=sum(r['string_null'] for r in cell),invalid=sum(not r['format_valid'] for r in cell),protocol_failure=sum(not r['protocol_valid'] for r in cell),target_correct=sum(r['target_correct'] for r in cell),auxiliary_correct=sum(r['auxiliary_correct'] is True for r in cell),n_aux=sum(r['auxiliary_correct'] is not None for r in cell)))
  rates={(t,f,c):statistics.mean(int(r['source_misuse']) for r in main if r['template_id']==t and r['form']==f and r['condition']==c) for t in templates for f in ['weak','direct'] for c in 'ABCD'}
  supported=[r for r in rs if r['form']=='supported'];support_rate=sum(r['target_correct'] for r in supported)/len(supported)
  support_by_condition={c:statistics.mean(int(r['target_correct']) for r in supported if r['condition']==c) for c in 'BD'}
  support_pass=support_rate>=threshold['min_supported_correct'] and all(v>=threshold['min_supported_per_condition'] for v in support_by_condition.values())
  protocol_failure_rate=statistics.mean(int(not r['protocol_valid']) for r in main)
  shares={}
  for form in ['weak','direct']:
   rr=[r for r in main if r['form']==form];nonnull=[r for r in rr if r['format_valid'] and r['target_category'] not in ['json_null','string_null']]
   source=sum(r['source_misuse'] for r in rr);shares[form]=dict(source=source,nonnull_nonencoding=len(nonnull),share=source/len(nonnull) if nonnull else 0)
  contrasts=[]
  for x,y in cfg['contrasts']:
   m={f:interval({t:rates[t,f,x]-rates[t,f,y] for t in templates}) for f in ['weak','direct']}
   m['interaction']=interval({t:(rates[t,'weak',x]-rates[t,'weak',y])-(rates[t,'direct',x]-rates[t,'direct',y]) for t in templates})
   field_forms=[f for f in ['weak','direct'] if abs(m[f]['mean'])>=threshold['min_abs_field_effect'] and m[f]['same_sign_templates']>=threshold['min_same_sign_templates']]
   qualified_forms=[f for f in field_forms if shares[f]['share']>=threshold['min_source_share']]
   inter=abs(m['interaction']['mean'])>=threshold['min_abs_interaction'] and m['interaction']['same_sign_templates']>=threshold['min_interaction_same_sign_templates']
   quality=support_pass and protocol_failure_rate<=threshold['max_protocol_failure']
   contrasts.append(dict(contrast=x+'-'+y,**m,field_effect_forms=field_forms,source_qualified_forms=qualified_forms,interaction_pass=inter,quality_pass=quality,full=bool(qualified_forms) and inter and quality,partial=bool(qualified_forms) and quality))
  full=[x['contrast'] for x in contrasts if x['full']];partial=[x['contrast'] for x in contrasts if x['partial']]
  label='full' if full else 'partial' if partial else 'not_replicated'
  result[family]=dict(label=label,full_contrasts=full,field_effect_contrasts=partial,cells=cells,contrasts=contrasts,source_shares=shares,supported_correct_rate=support_rate,supported_by_condition=support_by_condition,support_pass=support_pass,main_protocol_failure_rate=protocol_failure_rate)
 return dict(n=len(rows),counts=Counter(r['evidence'] for r in rows),families=result,n_full=sum(v['label']=='full' for v in result.values()),advance_to_formal=sum(v['label']=='full' for v in result.values())>=2,interpretation='Exploratory scope classification under locked operational criteria; not confirmatory significance or external validation.')

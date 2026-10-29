"""Template-cluster paired bootstrap, no independent treatment of paraphrase rows."""
import collections
import numpy as np
FORMS=['weak','direct','label','missing','supported']
METRICS=['office_city_misuse','launch_city_misuse','any_material_city_misuse','other_string','string_null','nonnull_output','reliable_null','supported_correct','false_null','auxiliary_correct','format_failure','protocol_failure']
def est(values,seed):
 x=np.array(list(values.values()),dtype=float)
 rng=np.random.default_rng(seed);boot=x[rng.integers(0,len(x),size=(2000,len(x)))].mean(axis=1)
 return dict(estimate=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n_templates=len(x),template_values=values)
def analyze(rows,config):
 seed=config['bootstrap']['seed'];templates=sorted({r['template_id'] for r in rows});cells=[];contrasts=[]
 def mean(t,f,c,o,metric):
  rr=[r for r in rows if r['template_id']==t and r['relation_form']==f and r['condition']==c and (o=='average' or c=='A' or r['target_order']==o)]
  return float(np.mean([r[metric] for r in rr]))
 for form in FORMS:
  for cond in ['A','B','C','D']:
   for order in (['only'] if cond=='A' else ['first','last','average']):
    rs=[r for r in rows if r['relation_form']==form and r['condition']==cond and (order=='average' or r['target_order']==order)]
    cell=dict(form=form,condition=cond,order=order,n_runs=len(rs),n_groups=len({r['group_id'] for r in rs}),n_templates=len(templates))
    for metric in METRICS:
     eligible=[r for r in rs if r[metric] is not None]
     if not eligible:continue
     values={t:float(np.mean([float(r[metric]) for r in eligible if r['template_id']==t])) for t in templates}
     cell[metric]={**est(values,seed),'numerator':sum(bool(r[metric]) for r in eligible),'denominator':len(eligible)}
    cells.append(cell)
  if form=='supported':continue
  for order in ['first','last','average']:
   for a,b in [('D','B'),('C','B'),('D','A'),('C','A'),('B','A')]:
    vals={t:mean(t,form,a,order,'office_city_misuse')-mean(t,form,b,order,'office_city_misuse') for t in templates}
    contrasts.append(dict(form=form,order=order,contrast=a+'-'+b,metric='office_city_misuse',**est(vals,seed)))
 for other in ['direct','label']:
  for order in ['first','last','average']:
   vals={t:(mean(t,'weak','D',order,'office_city_misuse')-mean(t,'weak','B',order,'office_city_misuse'))-(mean(t,other,'D',order,'office_city_misuse')-mean(t,other,'B',order,'office_city_misuse')) for t in templates}
   contrasts.append(dict(form='weak-minus-'+other,order=order,contrast='D-B',metric='office_city_misuse',**est(vals,seed)))
 primary=next(c for c in contrasts if c['form']=='weak' and c['order']=='average' and c['contrast']=='D-B')
 fmt=est({t:mean(t,'weak','D','average','protocol_failure')-mean(t,'weak','B','average','protocol_failure') for t in templates},seed)
 positive=[t for t,v in primary['template_values'].items() if v>0]
 local=(primary['estimate']>=config['gate']['min_primary'] and primary['ci95'][0]>0 and len(positive)>=config['gate']['min_positive_validation_templates'] and fmt['estimate']<config['gate']['max_format_excess'])
 return dict(split=rows[0]['split'],n_runs=len(rows),n_groups=len({r['group_id'] for r in rows}),n_templates=len(templates),cells=cells,contrasts=contrasts,
  categories=dict(collections.Counter(r['target_category'] for r in rows)),
  continuation=dict(primary=primary,positive_templates=positive,protocol_failure_excess=fmt,local_gate=bool(local),
   decision='requires_development_direction_check' if local else 'do_not_start_lens'))

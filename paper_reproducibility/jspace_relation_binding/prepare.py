"""Company relation attribution: new template-level holdout, frozen before inference."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SEED=20261016
CITIES=['武汉','上海','南京','成都','杭州','北京']
DEV={0,3,4,7,8,11}
# Each tuple is independently written: introduction/year, office variants,
# launch-event evidence, and a fixed block order. Variants remain within template.
TEMPLATES=[
 ('企业档案记载，{n}成立于{y}年。',
  '这家公司设有办公室，地点在{o}。','这家公司在{o}设有办公室。','办公室所在地：{o}。',
  '这家公司的一场产品发布会在{p}举办。','yol'),
 ('{n}在{y}年创立，以下是这家企业的业务资料。',
  '公司安排有一处办公室，该处地点为{o}。','公司在{o}安排有一处办公室。','办公城市一栏记为{o}。',
  '公司曾在{p}举行产品发布会。','ylo'),
 ('登记资料中，{n}的成立年份为{y}年。',
  '公司已经设立办公室，所在地为{o}。','公司已经在{o}设立办公室。','办公室地址中的城市：{o}。',
  '产品发布会的举办城市记作{p}。','oyl'),
 ('{n}的介绍写明：企业于{y}年成立。',
  '企业有一间日常使用的办公室，地点是{o}。','企业有一间位于{o}的日常办公室。','日常办公室所在城市为{o}。',
  '企业组织的产品发布会选择在{p}举行。','yol'),
 ('根据公司沿革，{n}成立的时间是{y}年。',
  '公司设立了一处办公室，这个地点在{o}。','公司设立了一处位于{o}的办公室。','办公地点记录：{o}。',
  '公司的一次产品发布会曾在{p}进行。','ylo'),
 ('关于{n}，现有记录包括其成立于{y}年。',
  '这家企业配有办公室，其位置在{o}。','这家企业配有位于{o}的办公室。','办公室位置对应的城市是{o}。',
  '记录还提到该企业在{p}举办过产品发布会。','oyl'),
 ('{n}自{y}年成立以来开展经营，资料列有以下事项。',
  '该企业有一处办公场所，地点为{o}。','该企业在{o}有一处办公场所。','办公场所的城市标注为{o}。',
  '该企业在{p}办过一场产品发布会。','yol'),
 ('{n}的公司简历将创立时间记为{y}年。',
  '公司启用了一间办公室，地址所在城市是{o}。','公司启用了设在{o}的一间办公室。','已启用办公室的城市：{o}。',
  '简历同时记载，公司产品发布会在{p}举办。','ylo'),
 ('资料摘录涉及{n}，其成立日期所属年份为{y}年。',
  '该公司拥有办公室，这处场所位于{o}。','该公司拥有位于{o}的办公室。','办公室所在城市记录为{o}。',
  '摘录中的另一项记录是该公司在{p}举办产品发布会。','oyl'),
 ('{n}的企业介绍确认了{y}年这一成立年份。',
  '企业开设了办公室，该处地址在{o}。','企业在{o}开设了办公室。','企业办公地址的城市一项为{o}。',
  '该介绍还写到企业在{p}举办产品发布会。','yol'),
 ('企业沿革显示，{n}于{y}年正式成立。',
  '这家公司有一间办公室，地点落在{o}。','这家公司有一间坐落在{o}的办公室。','该公司办公室的所在地是{o}。',
  '一场由该公司组织的产品发布会在{p}举行。','ylo'),
 ('{n}的成立时间在档案中写作{y}年。',
  '企业设有办公场所，该场所所在的城市为{o}。','企业在{o}设有办公场所。','企业办公场所所在地：{o}。',
  '档案也记下企业曾在{p}举办产品发布会。','oyl'),
 ('{n}成立于{y}年。','公司设有办公室，地点在{o}。','公司在{o}设有办公室。','办公城市：{o}。','公司曾在{p}举行产品发布会。','yol'),
 ('{n}于{y}年创立。','企业拥有办公室，地址在{o}。','企业在{o}拥有办公室。','办公室所在地记为{o}。','企业在{p}举办过产品发布会。','ylo')]
INSTRUCTION='任务：只依据下面材料抽取指定字段。不得利用外部知识、常识推测或其他字段的值补全缺失关系。证据不足时输出JSON的null，不要输出字符串“null”。仅输出一个JSON对象，不解释。'
DEFINITIONS='办公城市指材料明确说明设有办公室或办公场所的城市；总部城市指材料明确说明该公司总部所在的城市；发布会城市指该公司产品发布会举办的城市；成立年份指材料明确说明的成立年份。'
FORMS=['weak','direct','label','missing','supported']
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
 for d in ['data','results','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
 groups=[];jobs=[]
 for t,template in enumerate(TEMPLATES):
  split='pilot' if t>=12 else 'development' if t in DEV else 'validation'
  for j in range(1 if t>=12 else 3):
   n='澈岫珩棠朔沅璟霖砺沧栎筠泊桐'[t]+'禾舟岑'[j]+'实业公司'
   o=CITIES[(t*3+j)%6];p=CITIES[(t*3+j+3)%6];year=2002+(t*3+j)%22
   vals=dict(n=n,o=o,p=p,y=year);gid=f't{t+1:02d}_e{j+1}'
   intro,w,d,l,launch,order=template
   variants={}
   for form,office in [('weak',w),('direct',d),('label',l),('missing',d),('supported',d)]:
    if form in ['missing','supported'] and j!=0:continue
    # An explicit entity identifier makes office-first layouts unambiguous.
    parts={'y':intro.format(**vals),'o':office.format(**vals),'l':launch.format(**vals)}
    material=f'企业名称：{n}。'+''.join(parts[k] for k in order)
    if form=='missing':material+='该公司的总部所在地未披露。'
    if form=='supported':material+=f'该公司的总部位于{o}。'
    variants[form]=dict(text=material,gold={'总部城市':o if form=='supported' else None,'办公城市':o,'发布会城市':p,'成立年份':year})
   group=dict(group_id=gid,template_id=f'T{t+1:02d}',family='company',split=split,entity=n,office_city=o,launch_city=p,year=year,block_order=order,variants=variants)
   groups.append(group)
   for form,v in variants.items():
    for condition,aux in [('A',None),('B','办公城市'),('C','发布会城市'),('D','成立年份')]:
     for pos in (['only'] if aux is None else ['first','last']):
      fields=['总部城市'] if aux is None else ['总部城市',aux] if pos=='first' else [aux,'总部城市']
      prompt=INSTRUCTION+'\n字段定义：'+DEFINITIONS+'\n材料：'+v['text']+'\n本次需要输出的字段及顺序：'+'、'.join(fields)+'。'
      jobs.append(dict(job_id=f'{gid}_{form}_{condition}_{pos}',group_id=gid,template_id=group['template_id'],family='company',split=split,
       relation_form=form,evidence_version='supported' if form=='supported' else 'explicit_missing' if form=='missing' else 'absent',
       condition=condition,target_order=pos,fields=fields,target_field='总部城市',auxiliary_field=aux,
       candidate_city=o,office_city=o,launch_city=p,gold=v['gold'],material=v['text'],user_prompt=prompt))
 assert len(groups)==38 and len(jobs)==994
 assert len({g['entity'] for g in groups})==38
 assert not ({g['template_id'] for g in groups if g['split']=='development'}&{g['template_id'] for g in groups if g['split']=='validation'})
 for g in groups:
  assert g['office_city']!=g['launch_city']
  for f,v in g['variants'].items():
   assert v['text'].count(g['office_city'])==(2 if f=='supported' else 1)
   assert v['text'].count(g['launch_city'])==1
   assert v['gold']['总部城市']==(g['office_city'] if f=='supported' else None)
 config=dict(version='1.0',seed=SEED,model_path='models/Qwen3-8B',precision='bfloat16',attn_implementation='eager',batch_size=12,max_new_tokens=128,
  counts={s:sum(j['split']==s for j in jobs) for s in ['pilot','development','validation']},
  conditions={'A':'总部单字段','B':'办公城市+总部','C':'发布会城市+总部','D':'成立年份+总部'},
  forms={'weak':'办公室与城市分句连接','direct':'办公室与城市直接连接','label':'明确标注办公室城市','missing':'明确总部未披露（诊断）','supported':'明确总部在办公城市（正对照）'},
  template_split={s:sorted({g['template_id'] for g in groups if g['split']==s}) for s in ['pilot','development','validation']},
  primary='Validation weak-form D minus B in office-city-to-headquarters misattribution; equal weight of target-first/last, each template equal weight.',
  secondary=['Same-city-type C minus B','D minus A','C minus A','B minus A','Weak minus direct/label contrast interaction','Explicit missing and supported diagnostic controls kept separate'],
  normalization='Strip value whitespace only; cities must exactly match. Year integer or exact digit string allowed. No city aliases.',
  scoring='Separate office_city_misuse, launch_city_misuse, other_candidate (nonempty other string), string_null, true_null, missing/invalid, and strict protocol format issues. JSON string null never counts as a city.',
  bootstrap=dict(replicates=2000,seed=SEED,unit='template cluster retaining its three entities, all forms and field orders',ci='percentile 95%'),
  gate=dict(min_primary=.10,min_positive_validation_templates=4,require_ci_lower_positive=True,require_development_same_direction=True,max_format_excess=.05,
   interpretation='Passing weak-form gate supports at most weak-binding mechanism follow-up if direct/label do not replicate. No automatic claim of general field interference.'),
  pilot_acceptance='Valid JSON structure, exact keys/order/types and no truncation on every pilot row. String-null is separately logged as an instruction violation and does not trigger prompt tuning.',
  lens_policy='Only after template-held-out replication. Any city readout is a lexical representation, not evidence that its headquarters relation is supported. Prefix diagnostics separate from natural outputs.',
  human_review=dict(second_human_review=False,method='Handwritten template semantic review and exhaustive gold/city/variant invariants.'),
  limitations=['Six held-out templates give limited generalization and wide uncertainty.','Explicit label forms remain semantically similar despite distinct surrounding text.','Adding release-event evidence changes the context versus the old pilot; no direct causal old-versus-new comparison.','No real-world nonexistence guarantee for invented company names.'])
 save('protocol.json',config);save('data/groups.json',groups);save('data/jobs.json',jobs)
 # Preserve the old records as motivation only, never merge into new inference.
 oldpath=ROOT.parent/'jspace_multifield_interference/results/validation.jsonl'
 old=[json.loads(x) for x in oldpath.read_text(encoding='utf-8').splitlines()]
 old=[r for r in old if r['template_id']=='company_validation_4' and r['evidence_version']=='E1']
 save('data/prior_observation_audit.json',dict(source=str(oldpath),source_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),n=len(old),counts={c:dict(n=sum(r['condition']==c for r in old),office_city_misuse=sum(r['condition']==c and r['target_value']==r['candidate_city'] for r in old)) for c in ['A','B','C']},use='Motivation and read-only source audit; not new validation data.'))
 save('data/lock.json',{p:sha(p) for p in ['protocol.json','data/groups.json','data/jobs.json']})
 with (ROOT/'data/material_audit.csv').open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.writer(h);w.writerow(['group_id','template_id','split','form','text','headquarters_gold','office_city','launch_city','year'])
  for g in groups:
   for f,v in g['variants'].items():w.writerow([g['group_id'],g['template_id'],g['split'],f,v['text'],v['gold']['总部城市'],g['office_city'],g['launch_city'],g['year']])
 print(json.dumps(config['counts']))
if __name__=='__main__':main()

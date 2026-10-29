import json, hashlib, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for d in ['data','results','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
# New wording; no outcome-dependent material revision is permitted.
templates=[
 ('{n}的资料卡记有创办年份：{y}年。','公司租用了办公室，办公室所在城市是{o}。','公司租用了位于{o}的办公室。','资料卡另记公司总部位于{h}。','yo'),
 ('{n}的成立年份在沿革表中登记为{y}年。','企业保留着一间办公室，办公地点在{o}。','企业保留着一间位于{o}的办公室。','沿革表还记有企业总部，其所在城市为{h}。','oy'),
 ('{n}从{y}年开始成立运营。','公司新设了一间办公室，这间办公室的地址在{o}。','公司新设了一间地址在{o}的办公室。','公司总部的地址另列为{h}。','yo'),
 ('{n}的创立年份经记录确认为{y}年。','企业使用一处办公场所，这处办公场所位于{o}。','企业使用一处位于{o}的办公场所。','企业总部设在{h}。','oy'),
 ('{n}的公司档案把成立年份列为{y}年。','公司购置了办公室，办公室坐落于{o}。','公司购置了坐落于{o}的办公室。','档案中的总部所在地是{h}。','yo'),
 ('{n}于{y}年成立，这一信息收录在企业简表中。','企业另有一间办公室，这间办公室设在{o}。','企业另有一间设在{o}的办公室。','简表记录企业总部位于{h}。','oy'),
 ('{n}的成立年份为{y}年，以下为企业场所记录。','公司投入使用一处办公室，办公室的位置在{o}。','公司投入使用一处位于{o}的办公室。','场所记录注明公司总部在{h}。','yo'),
 ('{n}的登记摘要说明企业成立于{y}年。','企业配置了办公场所，该办公场所所在城市是{o}。','企业配置了位于{o}的办公场所。','登记摘要另将总部所在地记为{h}。','oy')]
instruction='任务：只依据下面材料抽取指定字段。不得利用外部知识、常识推测或其他字段的值补全缺失关系。证据不足时输出JSON的null，不要输出字符串“null”。仅输出一个JSON对象，不解释。'
definitions='办公城市指材料明确说明设有办公室或办公场所的城市；总部城市指材料明确说明该公司总部所在的城市；成立年份指材料明确说明的成立年份。'
jobs=[]
for t,(intro,weak,direct,hq,order) in enumerate(templates):
 split='development' if t<4 else 'validation'
 pairs=[('武汉','成都'),('成都','武汉'),('上海','南京'),('南京','上海')] if t<4 else [('武汉','上海'),('上海','武汉'),('成都','南京'),('南京','成都')]
 for e,(a,b) in enumerate(pairs):
  name='清晏砚禾松岚竹汀'[t]+'初远知明'[e]+'公司';year=2004+t*2+e
  for evidence in ['absent','supported','swapped']:
   office=b if evidence=='swapped' else a;head=a if evidence=='swapped' else b if evidence=='supported' else None
   for form,sentence in [('weak',weak),('direct',direct)]:
    v=dict(n=name,y=year,o=office,h=head);blocks={'y':intro.format(**v),'o':sentence.format(**v)}
    material=''.join(blocks[k] for k in order)+(hq.format(**v) if head else '')
    for condition,aux in [('B','办公城市'),('D','成立年份')]:
     fields=['总部城市',aux];gid=f'T{t+1:02d}_E{e+1}'
     jobs.append(dict(job_id=f'{gid}_{evidence}_{form}_{condition}',group_id=gid,template_id=f'T{t+1:02d}',split=split,evidence=evidence,form=form,condition=condition,city_a=a,city_b=b,office_city=office,hq_city=head,fields=fields,material=material,gold={'总部城市':head,aux:office if condition=='B' else year},user_prompt=instruction+'\n字段定义：'+definitions+'\n材料：'+material+'\n本次需要输出的字段及顺序：'+'、'.join(fields)+'。'))
assert len(jobs)==384 and len({j['job_id'] for j in jobs})==384
for i in range(0,len(jobs),2):assert jobs[i]['material']==jobs[i+1]['material']
save('data/jobs.json',jobs);save('data/templates.json',templates)
shutil.copyfile(ROOT.parent/'jspace_relation_recovery/data/prior_environment.json',ROOT/'data/prior_environment.json')
save('protocol.json',dict(model_path='models/Qwen3-8B',seed=20261015,batch_size=1,max_new_tokens=128,layers=[9,13,17,21,25,29,33],patch_layers=[29,33],ranks=[1,4],final_layer=35,
 split='T01-T04 development; T05-T08 validation; disjoint unordered city pairs, shared lexical city set; all new company names.',
 gate='Absent weak D-B >= .125; >=3/4 positive templates; direct D-B strictly smaller; supported+swapped HQ correctness >=.90 across B/D and both forms.',
 selection='Weak development PN: uncentered SVD of all 48 B-D pairs, all evidence; layers29/33 ranks1/4. Eligible if office misuse decreases and each supported stratum loses <=1 correct. Max absent null gain; tie rank then layer. If none, max macro three-stratum correctness gain, tie rank/layer.',
 validation='Only after selection.json or gate-fail decision is frozen. On fail no intervention, observational validation proceeds.',
 interventions='Weak PN only. Same-material B->D, reverse D->B, full B->D, norm-matched random B->D, self on both conditions. One-time prefill last-position patch; recipient cache thereafter.',
 missing_pn='Keep behavioral rows; report coverage. If any of 48 development weak pairs lacks PN, abort intervention rather than change SVD population. Missing validation pairs count as no improvement and fail success coverage.',
 random='One seeded orthonormal basis per selected layer/rank. Match projected perturbation norm per pair; zero intervention stays zero; numerically zero random projection uses first basis vector.',
 success='Validation absent null gain>=4/16, >=3/4 templates improve; supported and swapped each >=15/16, loss<=1; office->null transitions exceed format->null; null gain exceeds random; all48pairs available.',
 statistics='2000 template paired bootstrap for absent behavior interaction and intervention null gain only; descriptive, no p-values.',
 limits=['J-lens calibrated on English passages; readout quality on this task not guaranteed.','Subspace is field-requirement difference, not preidentified binding mechanism.','B donor is available at inference; this is causal mediation experiment, not deployable correction.','Natural PN may carry syntax differences; forced-prefix observation is auxiliary only.']))
save('data/lock.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['data/jobs.json','data/templates.json','data/prior_environment.json','protocol.json']})
print('Locked 384 jobs; 192 per split.')

from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent
for s in ['data','logs','results/qwen2b','results/qwen8b']:(R/s).mkdir(parents=True,exist_ok=True)
def save(p,x):(R/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
templates=[
('企业年鉴记载，{n}成立于{y}年。','{n}租有一间办公室，这间办公室的所在地是{o}。','{n}租有一间所在地为{o}的办公室。','年鉴另载，该公司的总部在{h}。','yoh'),
('在{n}的沿革记录中，成立时间记为{y}年。','{n}目前使用一间办公室，办公室位于{o}。','{n}目前使用一间位于{o}的办公室。','同一记录载明，公司总部位于{h}。','oyh'),
('{n}的基本资料包括以下内容：成立年份为{y}年。','该公司的办公室已经启用，其所在城市为{o}。','该公司在{o}的办公室已经启用。','该公司的总部位于{h}。','yho'),
('资料中的企业名称为{n}，成立年份是{y}年。','办公场所由该公司租赁，场所的位置在{o}。','该公司租赁了位于{o}的办公场所。','总部位置记载为{h}。','yoh'),
('{n}于{y}年创立。','走访记录提到该公司的办公室，记录中的办公城市为{o}。','走访记录提到该公司位于{o}的办公室。','记录另有说明：公司总部位于{h}。','yoh'),
('关于{n}，企业简介注明其创办于{y}年。','简介随后介绍了公司使用的办公地点，该地点在{o}。','简介随后介绍了公司在{o}使用的办公地点。','简介还注明公司总部设于{h}。','yho'),
('{n}的成立年份在年度报告中记为{y}年。','报告介绍了{n}的一处办公场所，这处场所位于{o}。','报告介绍了{n}位于{o}的一处办公场所。','报告记载的总部城市是{h}。','oyh'),
('企业信息表将{n}的成立年份填写为{y}年。','信息表有一项办公室记录，该办公室所在城市为{o}。','信息表记录了该公司设在{o}的办公室。','信息表将总部城市填写为{h}。','yoh'),
('{n}的资料分为沿革和场所两部分。沿革部分写明：公司成立于{y}年。','场所部分介绍了一间办公室，这间办公室的位置为{o}。','场所部分介绍了公司位于{o}的一间办公室。','场所部分另写明：总部位于{h}。','yoh'),
('档案中，{n}的创立时间为{y}年。','场所清单列有{n}的办公室，所在地记作{o}。','场所清单列有{n}设于{o}的办公室。','档案中，总部所在地记作{h}。','oyh'),
('{n}的企业简历写道：成立于{y}年。','办公设施一栏写道：公司设有办公室，该办公室位于{o}。','办公设施一栏写道：公司在{o}设有办公室。','总部信息一栏写道：总部位于{h}。','yho'),
('采访资料确认，{n}成立于{y}年。','资料还介绍了公司的办公场所，并将该办公场所的位置记为{o}。','资料还介绍了公司位于{o}的办公场所。','资料明确记载，公司总部坐落在{h}。','yoh'),
('{n}的沿革栏记有“{y}年成立”。','办公设施记录为“设有办公室，办公室地址在{o}”。','办公设施记录为“在{o}设有办公室”。','总部记录为“总部设在{h}”。','yho'),
('登记册对{n}作了如下记载：成立年份，{y}年。','{n}有一间自用办公室；这间办公室坐落在{o}。','{n}有一间坐落在{o}的自用办公室。','登记册另载：总部所在地，{h}。','oyh'),
('{n}的经营资料将企业成立时间追溯至{y}年。','资料列举了一处办公场所，所列办公场所在{o}。','资料列举了公司在{o}的一处办公场所。','资料另列公司总部，其所在地为{h}。','yoh'),
('{n}的资料摘要首先说明，企业成立于{y}年。','摘要接着说明，公司使用一处办公室，办公室所在城市是{o}。','摘要接着说明，公司使用一处设在{o}的办公室。','摘要还说明，公司总部设在{h}。','yho'),
('企业名录中的{n}，其成立年份为{y}年。','名录描述了公司的一处办公地点。该办公地点位于{o}。','名录描述了公司位于{o}的一处办公地点。','名录另将总部所在地列为{h}。','yoh'),
('{n}的成立年份经档案记载为{y}年。','{n}签约租赁了一间办公室。办公室的位置在{o}。','{n}签约租赁了一间位于{o}的办公室。','档案还记载，该公司的总部在{h}。','oyh'),
('根据企业资料，{n}于{y}年成立。','对于办公设施，资料称公司保有一间办公室，具体办公城市为{o}。','对于办公设施，资料称公司在{o}保有一间办公室。','对于总部，资料明确给出的城市为{h}。','yho'),
('企业介绍中，{n}的创办年份单独记为{y}年。','介绍另提及公司的办公室，并注明这间办公室设于{o}。','介绍另提及公司设于{o}的办公室。','介绍注明，公司总部所在城市为{h}。','yoh'),
('{n}的历史资料记载了企业的起点：{y}年成立。','资料也记载了{n}的办公室。其办公地址所在城市是{o}。','资料也记载了{n}位于{o}的办公室。','资料注明，其总部地址所在城市是{h}。','oyh'),
('{n}在企业索引中的成立年份是{y}年。','索引的场所说明提到，公司拥有办公室，办公室位于{o}。','索引的场所说明提到，公司在{o}拥有办公室。','索引的总部说明注明，总部位于{h}。','yho'),
('企业资料介绍了{n}的沿革，其中成立年份为{y}年。','另附的场所说明中有一间办公室，明确的办公地点为{o}。','另附的场所说明介绍了公司设在{o}的一间办公室。','公司总部则在资料中明确记为{h}。','yoh'),
('{n}的成立时间记录在企业履历中，年份为{y}年。','企业履历还提到{n}配置的一处办公场所，这处场所坐落于{o}。','企业履历还提到{n}配置在{o}的一处办公场所。','企业履历明确写出，总部所在地为{h}。','oyh')]
old=json.loads((P/'jspace_relation_mechanism_round1/data/jobs.json').read_text(encoding='utf-8'));prompt_head=old[0]['user_prompt'].split('\n材料：')[0]
oldnames={j['material'] for j in old};oldpairs={frozenset([j['city_a'],j['city_b']]) for j in old}
citypairs=[('武汉','南京'),('南京','武汉'),('上海','成都'),('成都','上海')]
assert not any(frozenset(c) in oldpairs for c in citypairs)
jobs=[];pairs=[]
for t,(intro,weak,direct,hq,order) in enumerate(templates):
 for e,(a,b) in enumerate(citypairs):
  name=['晟栖','沐衡','澜铎','珂原','栩泉','予珩','霁川','聿棠','呈砚','沅序','昕岳','祺舟','皓禾','铎岚','槿泉','颂原','栖远','承沐','宥川','璟禾','宜澜','珞舟','霖序','砚棠'][t]+['融','信','达','创'][e]+'公司';year=2001+(t*3+e)%22
  for evidence in ['absent','supported','swapped']:
   office=b if evidence=='swapped' else a;head=a if evidence=='swapped' else b if evidence=='supported' else None
   for form,sentence in [('weak',weak),('direct',direct)]:
    v=dict(n=name,y=year,o=office,h=head);blocks={'y':intro.format(**v),'o':sentence.format(**v),'h':hq.format(**v) if head else ''};material=''.join(blocks[k] for k in order); assert material not in oldnames
    pp={}
    for cond,aux in [('B','办公城市'),('D','成立年份')]:
     gid=f'X{t+1:02d}_E{e+1}';uid=f'{gid}_{evidence}_{form}_{cond}';fields=['总部城市',aux]
     j=dict(uid=uid,job_id=uid,group_id=gid,dataset='expanded',family='company',template_id=f'X{t+1:02d}',split='new_validation',form=form,evidence=evidence,target_order='first',target_field='总部城市',source_value=office,control_value=b if office==a else a,city_a=a,city_b=b,office_city=office,hq_city=head,condition=cond,fields=fields,material=material,gold={'总部城市':head,aux:office if cond=='B' else year},user_prompt=prompt_head+'\n材料：'+material+'\n本次需要输出的字段及顺序：'+'、'.join(fields)+'。')
     jobs.append(j);pp[cond]=j
    pairs.append(dict(pair_id=f'{gid}_{evidence}_{form}',**pp))
assert len(jobs)==1152 and len({j['material'] for j in jobs})==576
assert all(p['B']['material']==p['D']['material'] for p in pairs)
for t in range(1,25):
 for ev in ['absent','supported','swapped']:
  for form in ['weak','direct']:
   for cond in ['B','D']:assert sum(j['template_id']==f'X{t:02d}' and j['evidence']==ev and j['form']==form and j['condition']==cond for j in jobs)==4
save('data/jobs.json',jobs);save('data/pairs.json',pairs);save('data/templates.json',templates)
models={}
for key,path,layer,rank,gpu,runtime in [('qwen2b','jspace_multimodel_intervention/results/qwen2b',22,1,0,'./llm_knowledge/.venv/bin/python'),('qwen8b','jspace_relation_mechanism_round1/results',33,4,1,'python')]:
 for fname in ['frozen_basis.pt','random_basis.pt']:shutil.copyfile(P/path/fname,R/'data'/f'{key}_{fname}')
 env=json.loads((P/path/'environment.json').read_text(encoding='utf-8'))
 models[key]=dict(path='models/'+('Qwen3.5-2B' if key=='qwen2b' else 'Qwen3-8B'),layer=layer,rank=rank,gpu=gpu,runtime=runtime,original_results=path,expected_transformers=env['transformers'],basis_sha256=hashlib.sha256((R/'data'/f'{key}_frozen_basis.pt').read_bytes()).hexdigest())
save('protocol.json',dict(models=models,seed=20261026,max_new_tokens=128,precision='BF16',batch_size=1,decoding='greedy, native enable_thinking=False, eager attention',scope='24 new templates x 4 new companies/entity combinations x 3 evidence conditions x 2 expressions x B/D =1152 natural generations/model; 96 per cell. 24 template clusters, not1152 independent observations.',independence='No old template or company reused. City lexicon held fixed; unordered city pairs disjoint from original exploration/validation. No outcome-dependent edits.',intervention='Frozen original basis and random basis; no SVD or layer selection. Weak only: 288 B/D pairs/model. Natural B,D; matched-prefix replay B,D; self B,D; forward,reverse,full,random. Direct: natural B,D only.',boundary='Qwen3-8B: last exact opening-JSON colon boundary as original. Qwen3.5-2B: first natural target colon boundary as original. No fallback. Missing retained in denominators; no repair credited.',truncation='Fixed128 new tokens per call, retain truncation as failure, no long automatic retries.',scoring='Both strict JSON and one-complete-outer-fence-removed scores. Primary causal summary content scoring as figure8; primary behavioral summary strict. JSON null distinct from string null.',success='Descriptive extended criteria registered before execution: weak absent correct-null gain >=25pp; >=18/24 templates positive; supported and swapped each >=90/96 correct and loss <=6/96; source->null exceeds format->null; gain>one-seed random; all288 weak PN pairs available; self tokens identical. Report ceiling per template separately; no changed threshold after results.',statistics='2000 paired bootstrap samples over24 template clusters for primary behavior interaction and forward null gain only; template tables and transitions; no multiple significance tests.',report='New validation separate from old. No new J-lens calibration or scans in this behavioral/causal replication. Frozen checkpoints checked against historical hashes.'))
save('data/lock.json',{str(p.relative_to(R)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((R/'data').glob('*.json'))+list((R/'data').glob('*.pt'))+[R/'protocol.json'] if p.name!='lock.json'})
print('Locked 24 templates, 1152 natural jobs/model, 288 intervention pairs/model.')

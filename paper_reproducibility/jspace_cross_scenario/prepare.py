"""Four-domain scope pilot; handwritten templates, locked before any generation."""
import json, hashlib, shutil, csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
INSTRUCTION='任务：只依据下面材料抽取指定字段。不得利用外部知识、常识推测或其他字段的值补全缺失关系。证据不足时输出JSON的null，不要输出字符串“null”。仅输出一个JSON对象，不解释。'
# Each independently authored template: direct source relation, split source
# relation, same-type control, different-type control, information order.
FAMILIES={
 'person':dict(name='人物履历',fields=['出生城市','工作城市','毕业院校所在城市','入职年份'],numeric=True,
 definitions='出生城市指材料明确说明该人物出生的城市；工作城市指材料明确说明该人物工作的城市；毕业院校所在城市指材料明确说明其毕业院校所在的城市；入职年份指材料明确说明该人物入职的年份。工作、就学与出生关系不能互相替代。',
 missing='{n}的出生城市未披露。',supported='{n}出生于{h}。',
 templates=[
 ('{n}在{s}工作。','{n}有一份工作，工作地点在{s}。','{n}毕业于{org}，这所院校位于{c}。','{n}于{v}年入职。','svk'),
 ('履历记载，{n}的任职工作在{s}开展。','履历记载，{n}有任职工作，该工作的所在城市是{s}。','其毕业院校是坐落在{c}的{org}。','人事档案将其入职年份记为{v}年。','vks'),
 ('{n}目前在{s}从事日常工作。','{n}目前从事一份日常工作，这份工作的地点为{s}。','{n}毕业的{org}设在{c}。','这份履历注明{n}是在{v}年入职的。','ksv'),
 ('就任职情况而言，{n}在{s}工作。','就任职情况而言，{n}有工作安排，工作安排对应的城市为{s}。','学历一栏写着{n}毕业于{org}，院校所在城市为{c}。','入职记录确认了{v}年这一年份。','skv'),
 ('职业介绍说明{n}在{s}承担工作。','职业介绍说明{n}承担一份工作，该项工作的地点在{s}。','{n}的毕业院校{org}位于{c}。','{n}进入现工作单位的年份为{v}年。','kvs'),
 ('{n}的当前工作设在{s}。','{n}有一份当前工作，这份工作的城市是{s}。','教育经历显示，{n}毕业于位于{c}的{org}。','现工作单位记录其入职时间为{v}年。','vsk'),
 ('人员简介中，{n}在{s}工作任职。','人员简介中，{n}有一项工作任职，任职工作的地点记作{s}。','{n}曾从{org}毕业，该校所在城市是{c}。','简介还写明其于{v}年入职。','svk'),
 ('{n}的日常工作位于{s}。','{n}有日常工作，日常工作的所在地为{s}。','{org}是{n}的毕业院校，学校位于{c}。','关于入职，资料给出的年份是{v}年。','kvs')]),
 'event':dict(name='活动信息',fields=['本届活动举办城市','主办机构所在城市','往届活动举办城市','本届活动届数'],numeric=True,
 definitions='本届活动举办城市指材料明确说明本届活动举行的城市；主办机构所在城市指材料明确说明主办机构所在的城市；往届活动举办城市指材料提及的那一届往届活动举办的城市；本届活动届数指材料明确说明本届活动是第几届。机构地址、往届地点不能替代本届举办地点。',
 missing='{n}本届活动的举办城市未披露。',supported='{n}本届活动在{h}举行。',
 templates=[
 ('{n}本届活动的主办机构{org}位于{s}。','{n}本届活动有主办机构{org}，该机构所在城市为{s}。','{n}第{prev}届活动曾在{c}举行。','本届{n}是第{v}届。','svk'),
 ('位于{s}的{org}主办本届{n}。','本届{n}由{org}主办，这家主办机构的所在城市是{s}。','回顾资料记录，第{prev}届{n}在{c}举办。','活动编号显示，本届为第{v}届。','vks'),
 ('本届{n}由设在{s}的{org}主办。','本届{n}确定了主办机构{org}，主办机构设立的城市为{s}。','此前的第{prev}届活动在{c}举行。','本届{n}的届数为{v}。','ksv'),
 ('{n}本届的主办单位{org}坐落于{s}。','{n}本届安排了主办单位{org}，该单位所在的城市是{s}。','历史档案记载{n}第{prev}届的举办城市为{c}。','当前这届是{n}第{v}届活动。','skv'),
 ('设在{s}的{org}承担本届{n}的主办工作。','{org}承担本届{n}的主办工作，{org}的所在城市记为{s}。','{n}的一次往届记录为第{prev}届在{c}举办。','本届活动序号是第{v}届。','kvs'),
 ('本届{n}的主办方是位于{s}的{org}。','本届{n}设有主办方{org}，这一主办方所在的城市是{s}。','活动回顾写明第{prev}届{n}是在{c}举行的。','此次{n}属于第{v}届。','vsk'),
 ('{n}本届活动由驻于{s}的{org}主办。','{n}本届活动由{org}主办，主办机构的驻地城市为{s}。','第{prev}届{n}的举行地点在{c}。','本届活动为{n}第{v}届。','svk'),
 ('机构资料显示，本届{n}的主办机构{org}在{s}。','机构资料列出本届{n}的主办机构{org}，机构所在地一项记为{s}。','往届活动记录中，第{prev}届{n}的举办城市是{c}。','本届{n}标注的届次是第{v}届。','kvs')]),
 'product':dict(name='商品信息',fields=['产品生产城市','销售商所在城市','仓库所在城市','产品型号'],numeric=False,
 definitions='产品生产城市指材料明确说明该产品生产或制造的城市；销售商所在城市指材料明确说明该产品销售商所在的城市；仓库所在城市指材料明确说明存放该产品的仓库所在的城市；产品型号指材料明确给出的型号。销售、仓储与生产关系不能互相替代。',
 missing='{n}的生产城市未披露。',supported='{n}在{h}生产。',
 templates=[
 ('{n}的销售商{org}位于{s}。','{n}有销售商{org}，销售商所在城市为{s}。','存放{n}的仓库位于{c}。','{n}的产品型号是{v}。','svk'),
 ('位于{s}的{org}销售{n}。','{n}由{org}销售，这家销售商的所在城市是{s}。','{n}储存在设于{c}的仓库中。','产品登记将型号记为{v}。','vks'),
 ('{n}由设在{s}的销售商{org}经销。','{n}确定了销售商{org}，销售商设立的城市为{s}。','仓储记录表明，存放{n}的仓库在{c}。','{n}的型号标识为{v}。','ksv'),
 ('销售档案中，{n}的销售商{org}设于{s}。','销售档案中，{n}有销售商{org}，该销售商的所在地是{s}。','{n}对应的储存仓库坐落于{c}。','型号栏填写的是{v}。','skv'),
 ('设在{s}的{org}负责销售{n}。','{org}负责销售{n}，销售商{org}的所在城市记为{s}。','{n}存放在{c}的一处仓库。','{n}采用型号{v}。','kvs'),
 ('{n}的经销单位是位于{s}的{org}。','{n}有经销单位{org}，这家销售商所在的城市是{s}。','库存资料列明，{n}储存仓库的城市是{c}。','产品资料注明型号为{v}。','vsk'),
 ('{n}由驻于{s}的销售商{org}销售。','{n}由销售商{org}销售，销售商驻地所在的城市为{s}。','{n}入库存放的仓库位于{c}。','本产品的型号编号为{v}。','svk'),
 ('商品说明列出的销售商{org}位于{s}，负责销售{n}。','商品说明列出{n}的销售商{org}，销售商所在地一项记为{s}。','商品{n}的仓库所在城市记录为{c}。','商品说明为{n}标注了型号{v}。','kvs')]),
 'research':dict(name='科研成果',fields=['研究资助机构','作者所属机构','数据提供机构','发表年份'],numeric=True,
 definitions='研究资助机构指材料明确说明为该项研究提供经费资助的机构；作者所属机构指材料明确说明该成果作者任职或隶属的机构；数据提供机构指材料明确说明为该项研究提供数据的机构；发表年份指材料明确说明该成果发表的年份。任职、提供数据与提供经费资助是不同关系，不能互相替代。',
 missing='这项研究的经费资助机构未披露。',supported='这项研究的经费由{h}资助。',
 templates=[
 ('成果《{title}》的作者{n}任职于{s}。','成果《{title}》的作者{n}有任职单位，单位名称为{s}。','这项研究的数据由{c}提供。','成果《{title}》发表于{v}年。','svk'),
 ('{n}是《{title}》的作者，隶属于{s}。','《{title}》的作者是{n}，其所属机构的名称是{s}。','{c}向这项研究提供了数据。','该成果的发表年份记为{v}年。','vks'),
 ('任职于{s}的{n}撰写了成果《{title}》。','{n}撰写了成果《{title}》，作者任职单位一栏写着{s}。','研究所使用的数据来自提供方{c}。','《{title}》在{v}年发表。','ksv'),
 ('作者简介记载，《{title}》的作者{n}属于{s}。','作者简介记载，《{title}》由{n}撰写，作者所属单位为{s}。','这项研究接受了{c}提供的数据。','成果记录把发表时间记为{v}年。','skv'),
 ('{s}的工作人员{n}是成果《{title}》的作者。','成果《{title}》的作者{n}有工作单位，该任职单位为{s}。','数据说明确认{c}为本研究的数据提供机构。','这项成果于{v}年发表。','kvs'),
 ('成果《{title}》由在{s}任职的{n}撰写。','成果《{title}》由{n}撰写，其任职机构的名称记作{s}。','该研究的数据提供方是{c}。','文献记录注明发表年份为{v}年。','vsk'),
 ('《{title}》的作者{n}在{s}工作。','《{title}》的作者{n}有一处工作单位，工作单位名称为{s}。','为这项研究提供数据的是{c}。','《{title}》的发表时间属于{v}年。','svk'),
 ('作者资料显示，{n}任职于{s}，著有成果《{title}》。','作者资料显示，{n}著有成果《{title}》，其所属机构名称一项记为{s}。','本研究从{c}获得由该机构提供的数据。','发表记录显示该成果发表于{v}年。','kvs')])}
def main():
 for d in ['data','results','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
 cities=['武汉','上海','南京','成都','杭州','北京','长沙','合肥']
 institutions=['澄川研究院','泊岚研究中心','青岫实验室','栖原研究所','棠溪研究院','沅石研究中心','栎风实验室','霁禾研究所']
 surnames=['林','周','许','沈','顾','叶','苏','宋'];given=['沐岑','知遥','景禾','清越']
 jobs=[];materials=[]
 for fi,(family,cfg) in enumerate(FAMILIES.items()):
  for t,template in enumerate(cfg['templates']):
   for e in range(4):
    pool=institutions if family=='research' else cities
    s=pool[(t+e)%8];c=pool[(t+e+3)%8];h=pool[(t+e+5)%8]
    n=surnames[t]+given[e]
    if family=='event':n=['澄川','泊岚','青岫','栖原','棠溪','沅石','栎风','霁禾'][t]+['工艺交流会','设备展览会','设计论坛','读书节'][e]
    if family=='product':n=['澄川','泊岚','青岫','栖原','棠溪','沅石','栎风','霁禾'][t]+['台灯','测温仪','计时器','录音笔'][e]
    org=['泽棠','岚汀','青栎','沅林','砺禾','霁舟','泊珩','溪岑'][t]+['甲','乙','丙','丁'][e]+({'person':'学院','event':'文化中心','product':'贸易公司','research':'资料中心'}[family])
    v=5+t+e if family=='event' else f'Q{t+1:02d}-{e+1}K' if family=='product' else 2005+t+e
    values=dict(n=n,s=s,c=c,h=h,org=org,v=v,prev=v-2 if family=='event' else 0,title=['微光测量方法','纹理编码实验','声谱分析记录','材料表面观测'][e]+f'（{t+1}）')
    direct,weak,control,other,order=template;gid=f'{family}_t{t+1:02d}_e{e+1}'
    for form,source,suffix,target,conditions in [('direct',direct,'',None,'ABCD'),('weak',weak,'',None,'ABCD')]+([('missing',weak,cfg['missing'],None,'BD'),('supported',weak,cfg['supported'],h,'BD')] if e==0 else []):
     parts=dict(s=source,k=control,v=other);text=''.join(parts[k].format(**values) for k in order)+suffix.format(**values)
     fields=cfg['fields'];gold=dict(zip(fields,[target,s,c,v]));evidence='main' if form in ['direct','weak'] else form
     materials.append(dict(group_id=gid,family=family,template_id=f'{family}_T{t+1:02d}',form=form,text=text,gold=gold,source_value=s,control_value=c,supported_value=h))
     for condition in conditions:
      requested=[fields[0]] if condition=='A' else [fields[0],fields['ABCD'.index(condition)]]
      prompt=INSTRUCTION+'\n字段定义：'+cfg['definitions']+'\n材料：'+text+'\n本次需要输出的字段及顺序：'+'、'.join(requested)+'。'
      jobs.append(dict(job_id=f'{gid}_{form}_{condition}',group_id=gid,family=family,template_id=f'{family}_T{t+1:02d}',form=form,evidence=evidence,condition=condition,fields=requested,target_field=fields[0],numeric_field=fields[3] if cfg['numeric'] else None,source_value=s,control_value=c,supported_value=h,gold=gold,material=text,user_prompt=prompt))
 assert len(jobs)==1152 and sum(j['evidence']=='main' for j in jobs)==1024
 assert len({j['job_id'] for j in jobs})==1152
 for m in materials:
  assert len({m['source_value'],m['control_value'],m['supported_value']})==3
  assert m['text'].count(m['source_value'])==1 and m['text'].count(m['control_value'])==1
  assert m['text'].count(m['supported_value'])==(1 if m['form']=='supported' else 0)
 for g in {j['group_id'] for j in jobs}:
  for form in ['weak','direct']:
   js=[j for j in jobs if j['group_id']==g and j['form']==form]
   assert len(js)==4 and len({j['material'] for j in js})==1
 write('data/jobs.json',jobs);write('data/materials.json',materials)
 config=dict(seed=20261014,model_path='models/Qwen3-8B',batch_size=16,max_new_tokens=128,precision='bfloat16',attention='eager',
  scope='Exploratory cross-scenario pilot, all 32 templates. No confirmation/validation split; retain all scenarios. Fresh templates and entities required for formal study.',
  counts=dict(main=1024,diagnostic=128,total=1152,templates_per_family=8,entities_per_template=4),
  conditions=dict(A='target only',B='target+source',C='target+same-type control',D='target+different-type control'),
  families={k:dict(name=v['name'],fields=v['fields'],definitions=v['definitions']) for k,v in FAMILIES.items()},
  controls='First entity of all 8 templates per family, weak+explicit missing or weak+explicit supported different value, conditions B,D; 32 runs/family.',
  contrasts=[['B','A'],['C','A'],['D','A'],['C','B'],['D','B'],['D','C']],
  primary_metric='Exact source-value misattribution when target evidence absent. Structural-invalid outputs and string null are separate, not source errors. Denominator all planned rows.',
  interaction='(X-Y)weak minus (X-Y)direct, same entities paired, equal template weight; both positive and negative effects allowed.',
  discovery_rule=dict(min_abs_field_effect=.125,min_same_sign_templates=5,min_abs_interaction=.125,min_interaction_same_sign_templates=5,min_source_share=.60,min_supported_correct=15/16,min_supported_per_condition=7/8,max_protocol_failure=.05,
   meaning='For a full replication, the SAME prespecified contrast must pass field effect in at least one form AND expression interaction; positive and negative directions allowed. Source share among non-null, non-string-null valid target strings in the qualifying form must be >=60%. Supported controls must pass overall and per B/D. Partial replication passes a field effect and validity/source/support gates but no qualifying interaction. Error alone is not replication.',
   across_scenarios='Proceed to formal expansion only if >=2 new scenarios meet full replication. Other outcomes remain reported.',
   caveat='Operational exploratory thresholds, not significance tests. Six contrasts/family are screened without multiplicity-corrected inference. Intervals descriptive, not gate p-values.'),
  bootstrap=dict(replicates=2000,seed=20261014,unit='eight paired template clusters, retaining four entities and all conditions/forms',ci='percentile 95%'),
  limitations=['One model; target first only.','Generated fictional names do not guarantee real-world nonexistence.','Direct and split wording is a bundled syntax intervention, not a pure abstract clarity manipulation.','No second human annotation; handcrafted semantic review plus automated invariants.','Scope pilot may select favorable contrasts; formal preregistration needs new data.'])
 write('protocol.json',config)
 shutil.copyfile(ROOT.parent/'jspace_relation_binding/results/environment_main.json',ROOT/'data/prior_environment.json')
 write('data/lock.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['protocol.json','data/jobs.json','data/materials.json','prepare.py','run.py','scoring.py','analyze.py']})
 with (ROOT/'data/materials.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['family','template','group','form','material','gold']);w.writerows([m['family'],m['template_id'],m['group_id'],m['form'],m['text'],json.dumps(m['gold'],ensure_ascii=False)] for m in materials)
 print('Prepared and locked',len(jobs),'jobs')
if __name__=='__main__':main()

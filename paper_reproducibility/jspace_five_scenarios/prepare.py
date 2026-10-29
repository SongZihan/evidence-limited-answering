import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'jspace_cross_nohint'
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
INSTRUCTION='任务：只依据下面材料抽取指定字段。不得利用外部知识、常识推测或其他字段的值补全缺失关系。证据不足时输出JSON的null，不要输出字符串“null”。仅输出一个JSON对象，不解释。'
# Each template contains direct/split source relation, two fixed control facts,
# and a discourse order. These pairs are frozen before any model output.
FAMILIES={
 'shipment':dict(name='物流信息',fields=['收货城市','发货城市','中转城市','包裹重量'],numeric=True,kind='city',
 definitions='收货城市指材料明确说明包裹收件地址所在的城市；发货城市指材料明确说明包裹发出的城市；中转城市指材料明确说明包裹途中中转的城市；包裹重量指材料明确说明的重量，以千克为单位输出数值。',missing='该包裹的收货城市未披露。',supported='该包裹的收件地址位于{h}。',
 templates=[
 ('包裹{n}从{s}发出。','包裹{n}有发货记录，发货城市是{s}。','该包裹途中在{c}中转。','包裹重量为{v}千克。','svk'),
 ('物流单记载，{n}的发货地在{s}。','物流单记载，{n}有一处发货地，发货地所在城市为{s}。','这件包裹经过{c}的一次中转。','称重记录为{v}千克。','vks'),
 ('编号{n}的包裹在{s}交运发出。','编号{n}的包裹已经交运，交运发出城市为{s}。','运输记录注明这件包裹在{c}中转。','该包裹称重为{v}千克。','ksv'),
 ('运输档案显示包裹{n}由{s}发货。','运输档案显示包裹{n}已有发货记录，记录中的发货城市为{s}。','档案还记录该包裹在{c}中转。','重量一栏填写{v}千克。','skv'),
 ('包裹{n}的发出城市记为{s}。','包裹{n}设有发出地记录，发出地的城市记为{s}。','该包裹的一个中转城市为{c}。','该件货物重{v}千克。','kvs'),
 ('{n}这件包裹起运于{s}。','{n}这件包裹有起运记录，起运地点所在城市是{s}。','途中扫描记录表明该包裹在{c}中转。','包裹总重量为{v}千克。','vsk'),
 ('发货凭证显示包裹{n}从{s}寄出。','发货凭证显示包裹{n}已经寄出，寄出城市一项写着{s}。','转运信息列有该包裹在{c}的中转。','称重凭证上记着{v}千克。','svk'),
 ('包裹{n}的寄出地位于{s}。','包裹{n}登记了寄出地，寄出地对应的城市为{s}。','该包裹曾在{c}进行中转。','系统记录该包裹重量为{v}千克。','kvs')]),
 'film':dict(name='影视作品',fields=['故事发生城市','拍摄城市','首映城市','片长'],numeric=True,kind='city',
 definitions='故事发生城市指材料明确说明影片故事设定发生的城市；拍摄城市指材料明确说明影片进行实景拍摄的城市；首映城市指材料明确说明影片首映活动举办的城市；片长指材料明确说明的影片时长，以分钟为单位输出数值。',missing='这部影片的故事发生城市未披露。',supported='这部影片的故事设定发生在{h}。',
 templates=[
 ('影片《{n}》在{s}拍摄。','影片《{n}》有实景拍摄安排，拍摄城市是{s}。','这部影片的首映活动在{c}举办。','影片时长为{v}分钟。','svk'),
 ('《{n}》的实景拍摄地位于{s}。','《{n}》有一处实景拍摄地，拍摄地所在城市为{s}。','该片在{c}举行首映。','片长记为{v}分钟。','vks'),
 ('制作记录写明《{n}》在{s}取景拍摄。','制作记录写明《{n}》有取景拍摄工作，取景拍摄的城市为{s}。','首映资料记载该片的首映城市为{c}。','这部电影长{v}分钟。','ksv'),
 ('影片《{n}》的摄制在{s}进行。','影片《{n}》有摄制工作，摄制工作的地点在{s}。','该片的首映仪式设在{c}。','公映版本的片长为{v}分钟。','skv'),
 ('摄制档案显示，《{n}》的外景在{s}拍摄。','摄制档案显示，《{n}》安排了外景拍摄，外景拍摄城市为{s}。','档案记下该片在{c}举行过首映。','时长一栏标注{v}分钟。','kvs'),
 ('《{n}》曾在{s}开展实景拍摄。','《{n}》有实景拍摄记录，该拍摄记录中的城市是{s}。','影片首映礼的举办城市是{c}。','该片总时长{v}分钟。','vsk'),
 ('影片《{n}》在{s}完成取景拍摄。','影片《{n}》完成了取景拍摄，取景拍摄所在城市记作{s}。','这部影片于{c}举行首映礼。','资料给出的片长是{v}分钟。','svk'),
 ('影片《{n}》的实景镜头摄于{s}。','影片《{n}》有实景镜头拍摄，实景镜头拍摄城市为{s}。','首映活动记录中的举办城市为{c}。','影片介绍注明时长为{v}分钟。','kvs')]),
 'artifact':dict(name='文物档案',fields=['出土城市','展出城市','修复城市','入藏年份'],numeric=True,kind='city',
 definitions='出土城市指材料明确说明文物发掘出土的城市；展出城市指材料明确说明文物进行展览的城市；修复城市指材料明确说明文物接受修复的城市；入藏年份指材料明确说明文物被收入馆藏的年份。',missing='这件文物的出土城市未披露。',supported='这件文物在{h}发掘出土。',
 templates=[
 ('文物{n}在{s}展出。','文物{n}有展览安排，展出城市是{s}。','这件文物曾在{c}接受修复。','文物于{v}年入藏。','svk'),
 ('编号{n}的文物展览在{s}进行。','编号{n}的文物参加过展览，该次展览的城市为{s}。','该文物的修复工作在{c}完成。','馆藏记录中的入藏年份为{v}年。','vks'),
 ('展览资料显示文物{n}曾陈列于{s}。','展览资料显示文物{n}有陈列记录，展览陈列的城市为{s}。','修复档案记下该文物在{c}修复。','这件文物在{v}年收入馆藏。','ksv'),
 ('文物{n}在{s}公开展览。','文物{n}有一次公开展览，展览所在城市是{s}。','这件文物的修复地点在{c}。','入藏记录写明时间为{v}年。','skv'),
 ('文物{n}的展出地点设在{s}。','文物{n}设有展出地点，展出地点所在城市为{s}。','该文物曾送至{c}完成修复。','文物档案将入藏时间记为{v}年。','kvs'),
 ('馆方资料记载文物{n}于{s}展出。','馆方资料记载文物{n}已有展出记录，记录中的展出城市为{s}。','修复记录中的工作城市是{c}。','该文物自{v}年起收入馆藏。','vsk'),
 ('文物{n}曾在{s}参加展览。','文物{n}曾参加一次展览，展览城市一栏为{s}。','该件文物在{c}进行过修复处理。','文物的入藏年份为{v}年。','svk'),
 ('文物{n}的一次展览位于{s}。','文物{n}有一次展览经历，展览所在的城市记为{s}。','这件文物的修复城市记录为{c}。','馆藏登记注明它于{v}年入藏。','kvs')]),
 'book':dict(name='图书出版',fields=['著作权持有机构','出版机构','印刷机构','出版年份'],numeric=True,kind='org',
 definitions='著作权持有机构指材料明确说明持有该书著作权的机构；出版机构指材料明确说明负责出版该书的机构；印刷机构指材料明确说明承担该书印刷工作的机构；出版年份指材料明确说明该书出版的年份。',missing='该书的著作权持有机构未披露。',supported='该书的著作权由{h}持有。',
 templates=[
 ('图书《{n}》由{s}出版。','图书《{n}》有出版机构，机构名称为{s}。','该书由{c}承担印刷。','该书出版于{v}年。','svk'),
 ('{s}负责图书《{n}》的出版。','图书《{n}》有负责出版的单位，该出版单位是{s}。','这本书的印刷机构为{c}。','出版记录标注的年份为{v}年。','vks'),
 ('出版资料记载《{n}》由{s}出版。','出版资料为《{n}》列出了出版机构，出版机构名称记为{s}。','该书的印制工作由{c}完成。','《{n}》于{v}年出版。','ksv'),
 ('《{n}》的出版单位为{s}。','《{n}》有一个出版单位，单位名称一栏写着{s}。','印刷资料列明承印机构是{c}。','这本书的出版年份是{v}年。','skv'),
 ('图书《{n}》交由{s}出版。','图书《{n}》确定了出版机构，该机构为{s}。','{c}承印了这本书。','书目记录的出版时间属于{v}年。','kvs'),
 ('图书《{n}》的出版工作由{s}负责。','图书《{n}》安排了出版工作，承担出版的机构是{s}。','图书的印刷工作交由{c}承担。','该书在{v}年出版发行。','vsk'),
 ('{s}出版了图书《{n}》。','图书《{n}》已确定出版单位，出版单位名称为{s}。','该书的印刷单位登记为{c}。','资料注明该书出版于{v}年。','svk'),
 ('书目档案显示《{n}》由{s}出版发行。','书目档案列有《{n}》的出版机构，机构名称一项填写{s}。','这本书由{c}印制。','出版年份一栏记录为{v}年。','kvs')]),
 'procurement':dict(name='项目采购',fields=['合同签订日期','中标日期','招标公告日期','项目编号'],numeric=False,kind='date',
 definitions='合同签订日期指材料明确说明该项目合同实际签订的日期；中标日期指材料明确说明该项目确定中标结果的日期；招标公告日期指材料明确说明该项目招标公告发布的日期；项目编号指材料明确给出的编号。日期字段统一以YYYY-MM-DD格式输出。',missing='该项目的合同签订日期未披露。',supported='该项目的合同于{h}签订。',
 templates=[
 ('{n}于{s}确定中标结果。','{n}有中标结果记录，确定中标结果的日期是{s}。','该项目的招标公告于{c}发布。','项目编号为{v}。','svk'),
 ('{n}的中标结果在{s}确定。','{n}已有中标结果，中标日期一栏为{s}。','项目招标公告的发布日期是{c}。','该项目登记编号为{v}。','vks'),
 ('采购记录显示，{n}在{s}确定中标方。','采购记录显示，{n}有中标方确定记录，对应的中标日期为{s}。','该项目于{c}刊发招标公告。','项目的编号标记为{v}。','ksv'),
 ('{n}的中标日期记为{s}。','{n}设有中标日期记录，该日期记作{s}。','招标公告记录显示发布日期为{c}。','采购档案列出的项目编号是{v}。','skv'),
 ('{n}在{s}完成中标结果确定。','{n}完成了中标结果确定，中标日期记录为{s}。','该项目招标公告在{c}发布。','项目采用编号{v}。','kvs'),
 ('项目档案确认{n}于{s}确定中标结果。','项目档案列出{n}的中标结果，确定该结果的日期是{s}。','该项目的招标公告发布日期记为{c}。','档案中的项目编号为{v}。','vsk'),
 ('{n}在{s}确定了中标单位。','{n}有中标单位确定记录，确定日期一项写着{s}。','项目曾于{c}发布招标公告。','项目识别编号是{v}。','svk'),
 ('{n}的中标结果确定日期为{s}。','{n}记录了中标结果确定事项，该事项的日期为{s}。','本项目招标公告刊发日期为{c}。','项目编号一栏填写{v}。','kvs')])}
def main():
 for d in ['data','results','logs']:(ROOT/d).mkdir(parents=True,exist_ok=True)
 cities=['武汉','上海','南京','成都','杭州','北京','长沙','合肥'];orgs=['澄页文化公司','泊文文化公司','青简文化公司','栖墨文化公司','棠序文化公司','沅书文化公司','栎章文化公司','霁卷文化公司'];jobs=[];materials=[]
 for f,cfg in FAMILIES.items():
  for t,tpl in enumerate(cfg['templates']):
   for e in range(4):
    i=(t+e)%8
    if cfg['kind']=='date':s=f'2024-03-{10+i:02d}';c=f'2024-03-{1+i:02d}';h=f'2024-03-{20+i:02d}'
    else:
     pool=orgs if cfg['kind']=='org' else cities;s=pool[i];c=pool[(i+3)%8];h=pool[(i+5)%8]
    n={'shipment':f'LK{t+1:02d}{e+1}','film':['潮汐之路','山间来信','远岸微光','午后回声'][e]+f'·{t+1}','artifact':f'藏品WH{t+1:02d}{e+1}','book':['纸上群山','静水札记','风物索引','远行手稿'][e]+f'·{t+1}','procurement':f'园区设备采购项目{t+1:02d}-{e+1}'}[f]
    v={'shipment':2+t+e,'film':80+t*3+e,'artifact':2005+t+e,'book':2010+t+e,'procurement':f'CG-{t+1:02d}-{e+1}X'}[f]
    vals=dict(n=n,s=s,c=c,h=h,v=v);direct,weak,control,other,order=tpl;gid=f'{f}_t{t+1:02d}_e{e+1}'
    variants=[('weak',weak,'',None,'ABCD'),('direct',direct,'',None,'ABCD')]
    if e==0:variants += [('missing',weak,cfg['missing'],None,'BD'),('supported',weak,cfg['supported'],h,'BD')]
    for form,src,suffix,target,conds in variants:
     parts=dict(s=src,k=control,v=other);text=''.join(parts[k].format(**vals) for k in order)+suffix.format(**vals);fields=cfg['fields'];gold=dict(zip(fields,[target,s,c,v]));evidence='main' if form in ['weak','direct'] else form
     assert len({s,c,h})==3 and text.count(s)==1 and text.count(c)==1 and text.count(h)==(form=='supported')
     materials.append(dict(group_id=gid,family=f,template_id=f'{f}_T{t+1:02d}',form=form,text=text,gold=gold))
     for cond in conds:
      requested=[fields[0]] if cond=='A' else [fields[0],fields['ABCD'.index(cond)]]
      prompt=INSTRUCTION+'\n字段定义：'+cfg['definitions']+'\n材料：'+text+'\n本次需要输出的字段及顺序：'+'、'.join(requested)+'。'
      assert '不能互相替代' not in prompt and '不能替代' not in prompt
      jobs.append(dict(job_id=f'{gid}_{form}_{cond}',group_id=gid,family=f,template_id=f'{f}_T{t+1:02d}',form=form,evidence=evidence,condition=cond,fields=requested,target_field=fields[0],numeric_field=fields[3] if cfg['numeric'] else None,source_value=s,control_value=c,supported_value=h,gold=gold,material=text,user_prompt=prompt))
 assert len(jobs)==1440 and len({j['job_id'] for j in jobs})==1440
 for gid in {j['group_id'] for j in jobs}:
  for form in ['weak','direct']:
   xs=[j for j in jobs if j['group_id']==gid and j['form']==form];assert len(xs)==4 and len({j['material'] for j in xs})==1
 save('data/jobs.json',jobs);save('data/materials.json',materials)
 cfg=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'));cfg.pop('paired_rerun',None)
 cfg.update(scope='Five NEW exploratory scenarios, 40 templates, no confirmatory split. Same no-hint instruction framework and discovery thresholds as previous pilot.',counts=dict(main=1280,diagnostic=160,total=1440,templates_per_family=8,entities_per_template=4),families={k:dict(name=v['name'],fields=v['fields'],definitions=v['definitions']) for k,v in FAMILIES.items()})
 cfg['limitations']+=['Procurement dates use chronological announcement < award < signing controls, but missing signing date is never inferred.','Date output uses YYYY-MM-DD; alternate representations retained as other strings, not silently normalized.','This expands an already explored scenario search; thresholds are screening rules, not corrected significance tests.']
 save('protocol.json',cfg)
 for name in ['run.py','scoring.py','analyze.py']:shutil.copyfile(OLD/name,ROOT/name)
 shutil.copyfile(OLD/'data/prior_environment.json',ROOT/'data/prior_environment.json')
 save('data/lock.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['prepare.py','protocol.json','data/jobs.json','data/materials.json','run.py','scoring.py','analyze.py']})
 print('Locked 1440 jobs, 5 scenarios, 40 templates; no domain-specific warning.')
if __name__=='__main__':main()

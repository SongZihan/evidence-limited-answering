import json,html,re,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
NAMES=dict(company='公司总部',person='人物出生地',event='活动举办地',product='产品生产地',research='研究资助机构',shipment='包裹收货地',film='故事发生地',artifact='文物出土地',book='著作权持有机构',procurement='合同签订日期')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,x):(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','／').replace('\n',' ') for x in row)+' |' for row in data])
def lookup(p,kind,condition=None):return next((r for r in p['rows'] if r['kind']==kind and (condition is None or r['uid'].endswith('_'+condition) or ('_'+condition+'_') in r['uid'])),None)
def stats(ps,kind):
 s=dict(planned=len(ps),available=0,base_correct=0,after_correct=0,base_source=0,after_source=0,base_null=0,after_null=0,repair=0,harm=0,source_repair=0,base_invalid=0,after_invalid=0,transitions=Counter())
 for p in ps:
  if not p['available']:continue
  result=lookup(p,kind);base=next(r for r in p['rows'] if r['kind']=='replay' and r['uid']==result['uid'])
  s['available']+=1;s['base_correct']+=base['correct'];s['after_correct']+=result['correct'];s['base_source']+=base['source_misuse'];s['after_source']+=result['source_misuse'];s['base_null']+=base['true_null'];s['after_null']+=result['true_null'];s['repair']+=not base['correct'] and result['correct'];s['harm']+=base['correct'] and not result['correct'];s['source_repair']+=base['source_misuse'] and result['correct'];s['base_invalid']+=not base['format_valid'];s['after_invalid']+=not result['format_valid'];s['transitions'][base['category']+' → '+result['category']]+=1
 s['net']=s['after_correct']-s['base_correct'];return s
def main():
 statuses=[read(R/f'status_{i}.json') for i in [0,1]];assert all(s['stage']=='complete' for s in statuses)
 pairs=[json.loads(line) for i in [0,1] for line in (R/f'pairs_{i}.jsonl').read_text(encoding='utf-8').splitlines()]
 planned=read(ROOT/'data/pairs.json');byid={p['pair_id']:p for p in planned}
 assert len(pairs)==1176 and len({p['pair_id'] for p in pairs})==1176 and set(byid)=={p['pair_id'] for p in pairs}
 allrows=[r for p in pairs for r in p['rows']];selfcount=0;normdrift=[]
 for p in pairs:
  assert sum(r['kind']=='natural' for r in p['rows'])==2
  if p['available']:
   assert len(p['rows'])==10
   for r in p['rows']:
    if r['kind'].startswith('self'):
     b=next(x for x in p['rows'] if x['kind']=='replay' and x['uid']==r['uid']);assert r['generated_token_ids']==b['generated_token_ids'];selfcount+=1
   forward=lookup(p,'forward');random=lookup(p,'random')
   assert abs(forward['intended_delta_norm']-random['intended_delta_norm'])<=max(1e-4,forward['intended_delta_norm']*1e-5)
   if forward['realized_delta_norm']>0:normdrift.append(abs(forward['realized_delta_norm']-random['realized_delta_norm'])/forward['realized_delta_norm'])
 audit=dict(pairs=len(pairs),available=sum(p['available'] for p in pairs),natural=sum(r['kind']=='natural' for r in allrows),replay=sum(r['kind']=='replay' for r in allrows),self_checks=selfcount,generations=len(allrows),natural_replay_differences=sum(not r['natural_equal'] for r in allrows if r['kind']=='replay'),truncated=sum(r['truncated'] for r in allrows),natural_invalid=sum(not r['format_valid'] for r in allrows if r['kind']=='natural'),random_intended_norm_match=True,realized_norm_relative_difference_mean=sum(normdrift)/len(normdrift) if normdrift else None,realized_norm_relative_difference_max=max(normdrift) if normdrift else None)
 audit['aux_errors_by_kind']={k:dict(n=sum(r['kind']==k for r in allrows),errors=sum(not r['aux_correct'] for r in allrows if r['kind']==k)) for k in ['natural','replay','forward','reverse','full','random']}
 save('completion_audit.json',audit)
 kinds=['forward','random','full','reverse'];summary=[]
 for ds in ['company','four','five','round1']:
  for family in sorted({p['family'] for p in pairs if p['dataset']==ds}):
   groups=sorted({(p['split'],p['form'],p['evidence'],p['target_order']) for p in pairs if p['dataset']==ds and p['family']==family})
   for split,form,evidence,order in groups:
    ps=[p for p in pairs if (p['dataset'],p['family'],p['split'],p['form'],p['evidence'],p['target_order'])==(ds,family,split,form,evidence,order)]
    for kind in kinds:summary.append(dict(dataset=ds,family=family,split=split,form=form,evidence=evidence,order=order,kind=kind,**stats(ps,kind)))
 save('stratified_summary.json',summary)
 template=[]
 for key in sorted({(p['dataset'],p['family'],p['template_id'],p['split'],p['form'],p['evidence'],p['target_order']) for p in pairs}):
  ps=[p for p in pairs if (p['dataset'],p['family'],p['template_id'],p['split'],p['form'],p['evidence'],p['target_order'])==key]
  template.append(dict(zip(['dataset','family','template_id','split','form','evidence','order'],key),**stats(ps,'forward')))
 save('template_summary.json',template)
 historical=[p for p in pairs if p['dataset']!='round1'];mainrows=[];maindata=[];diagdata=[]
 for family in NAMES:
  ps=[p for p in historical if p['family']==family and p['evidence']=='main'];ss={k:stats(ps,k) for k in kinds};mainrows.append(dict(family=family,**ss));f=ss['forward'];rand=ss['random']
  pertemplate=[stats([p for p in ps if p['template_id']==t],'forward') for t in sorted({p['template_id'] for p in ps})]
  mainrows[-1]['template_coverage']=dict(total=len(pertemplate),improved=sum(t['net']>0 for t in pertemplate),with_baseline_error=sum(t['base_correct']<t['available'] for t in pertemplate),harmed=sum(t['harm']>0 for t in pertemplate))
  maindata.append([NAMES[family],f"{f['available']}/{f['planned']}",f"{f['base_source']}→{f['after_source']}",f"{f['base_correct']}→{f['after_correct']}",f['repair'],f['harm'],f"{sum(t['net']>0 for t in pertemplate)}/{len(pertemplate)}",rand['net'],ss['full']['net'],ss['reverse']['net']])
  for diagnostic in ['supported','explicit_missing','missing']:
   dp=[p for p in historical if p['family']==family and p['evidence']==diagnostic]
   if not dp:continue
   d=stats(dp,'forward');natural_correct=sum(next(r for r in p['rows'] if r['kind']=='natural' and r['uid']==byid[p['pair_id']]['D']['uid'])['correct'] for p in dp)
   diagdata.append([NAMES[family],diagnostic,f"{natural_correct}/{len(dp)}",f"{d['available']}/{d['planned']}",f"{d['base_correct']}→{d['after_correct']}",f"{d['base_null']}→{d['after_null']}",d['harm']])
 save('main_summary.json',mainrows)
 chunks=['# 冻结4维子空间的十场景迁移实验','复用第一轮公司任务选择的第33层、4维子空间，不重新拟合、不更换方向、不筛选原本会出错的例子。模型仍为 Qwen3-8B。历史十场景与第一轮补齐结果分开报告。','## 1. 覆盖范围与统计口径',
 '历史3516个提示位置覆盖852份不同正文；同一正文按字段要求与顺序形成984对 B/D 配对。另补第一轮192对，共1176对、1044份不同正文。A/C提示使用的正文均已覆盖，但本轮干预定义为B与D之间的状态转移，因此不另执行A/C条件干预。',
 'B、D的含义按原场景保留：B要求目标字段＋来源字段，D要求目标字段＋不同类型字段；公司历史任务同时包括目标前置与后置。每对重跑自然生成和目标值前的重放。正向为B→D，反向为D→B；完整状态和固定随机子空间都使用同事实、同证据供体。',
 '**因果比较均以本轮重放为基线，不直接与旧批量生成作差。** 修复＝原本错误变正确；破坏＝原本正确变错误；净增＝修复减破坏。来源误填只是全部错误的一部分。没有证据的普通材料、明确缺失诊断、有充分证据诊断分开；历史数据不是新的独立验证。',
 '## 2. 历史十场景主实验',table(['场景','可干预/计划配对','来源误填 前→后','目标正确 前→后','修复','破坏','净改善模板/全部模板','随机净增','完整替换净增','反向净增'],maindata),
 '表中主干预为冻结4维B→D；反向列的基线是B，其他列的基线是D。各场景沿用原材料表达与字段顺序的数量，不能用未分层的总数直接比较任务难度。公司场景包含label表达及两种字段顺序，其余场景为weak/direct且目标前置。',
 '## 3. 明确缺失与充分证据诊断',table(['场景','诊断','自然D正确/全部','可干预/计划配对','目标正确 前→后','null 前→后','破坏'],diagdata),
 '自然D正确率覆盖全部计划样本；前后比较仅包括两侧都能定位自然PN的配对。无法定位不等于回答错误，也不能当作干预无效或有效。',
 '历史公司充分证据诊断中，总部城市与办公城市相同，正确也可能是误用了来源值；其区分力弱于其他场景的不同目标值诊断。第一轮补齐部分另有总部与办公室不同及关系交换条件。',
 '## 4. 表达形式、顺序及数据来源分层']
 data=[]
 for s in summary:
  if s['dataset']=='round1' or s['kind']!='forward':continue
  data.append([NAMES[s['family']],s['split'],s['form'],s['order'],s['available'],f"{s['base_correct']}→{s['after_correct']}",s['repair'],s['harm']])
 chunks.append(table(['场景','原划分','表达/证据形式','目标顺序','配对数','正确 前→后','修复','破坏'],data))
 chunks.append('## 5. 第一轮全部材料补齐（非新增独立验证）')
 data=[]
 for s in summary:
  if s['dataset']!='round1' or s['kind']!='forward':continue
  data.append([s['split'],s['evidence'],s['form'],s['available'],f"{s['base_correct']}→{s['after_correct']}",s['repair'],s['harm']])
 chunks+= [table(['原划分','证据','表达','配对数','正确 前→后','修复','破坏'],data),'探索数据参与过子空间拟合；第一轮验证数据已被查看过。本次补齐不能重新包装成独立验证，也不改变上一轮未通过3/4模板门槛的结论。','## 6. 代表性错误转移']
 case_rows=[]
 for family in NAMES:
  for label in ['repair','harm']:
   options=[]
   for p in historical:
    if p['family']!=family or not p['available']:continue
    a=lookup(p,'forward');b=next(r for r in p['rows'] if r['kind']=='replay' and r['uid']==a['uid'])
    if (label=='repair' and not b['correct'] and a['correct']) or (label=='harm' and b['correct'] and not a['correct']):options.append((p,b,a))
   if options:
    p,b,a=sorted(options,key=lambda x:x[0]['pair_id'])[0];case_rows.append([NAMES[family],label,p['pair_id'],byid[p['pair_id']]['D']['material'],b['raw_output'],a['raw_output']])
 chunks.append(table(['场景','类型','配对编号','正文','重放输出','4维干预输出'],case_rows) if case_rows else '没有目标正确性发生变化的配对。')
 unavailable=[dict(pair_id=p['pair_id'],available_sides=list(p['boundaries']),natural=[r for r in p['rows'] if r['kind']=='natural']) for p in pairs if not p['available']]
 save('unavailable_pairs.json',unavailable)
 chunks+=['## 7. 技术检查与局限',
 f"计划{audit['pairs']}对，可定位两侧目标值前位置{audit['available']}对；自然生成{audit['natural']}条、重放{audit['replay']}条、全部生成{audit['generations']}次。自替换{audit['self_checks']}次均与对应重放token完全一致。自然与重放续写不同{audit['natural_replay_differences']}条。全部生成中的截断{audit['truncated']}条，自然格式失败{audit['natural_invalid']}条。",
 '未进入干预的配对详见[位置覆盖记录](unavailable_pairs.json)。紧凑JSON可能将冒号和起始引号合在同一token中，没有计划所需的、值开始前的独立自然token边界；不拆分token或改用强制前缀来补齐，以免改变干预定义。',
 '随机对照沿用第一轮固定随机基，逐例匹配写入前的浮点扰动范数；模型以BF16接收状态，量化后范数可能略有不同。其相对范数差异均值／最大值为 '+str(audit['realized_norm_relative_difference_mean'])+' / '+str(audit['realized_norm_relative_difference_max'])+'。只有一个随机基，不据此作普遍显著性结论。',
 '以上正确数均为目标字段正确数，不等于两个字段全部正确；辅助字段错误另列：'+json.dumps(audit['aux_errors_by_kind'],ensure_ascii=False)+'。其中格式失败也不能视作辅助字段正确。',
 '目标值前的状态同时可能含关系、答案倾向、字段名称及已生成内容。本实验检验冻结子空间能否跨任务改变输出，不单独证明关系绑定机制。目标后置时，B与D前面已经生成的字段内容不同，必须与前置结果分开解释。低维转移成功也不代表可以不用同事实B供体直接部署修复。',
 '## 8. 复查文件',
 '[协议](../protocol.json) · [全部配对与提示词](../data/pairs.json) · [材料覆盖审计](../data/coverage.json) · [冻结配置来源](../data/original_selection.json) · [主实验汇总](main_summary.json) · [完整分层](stratified_summary.json) · [逐模板](template_summary.json) · [完整性检查](completion_audit.json) · [GPU0逐条输出](pairs_0.jsonl) · [GPU1逐条输出](pairs_1.jsonl)']
 positive=[r for r in mainrows if r['forward']['net']>0];floors=[r for r in mainrows if r['forward']['base_correct']==r['forward']['available'] and r['forward']['after_correct']==r['forward']['available']]
 supported_stats=stats([p for p in historical if p['gold'] is not None],'forward')
 overview='历史主实验中，冻结4维替换共修复'+str(sum(r['forward']['repair'] for r in mainrows))+'个原有目标错误，破坏'+str(sum(r['forward']['harm'] for r in mainrows))+'个原本正确的目标答案；同范数随机对照净改善'+str(sum(r['random']['net'] for r in mainrows))+'个。正面修复集中在'+('、'.join(NAMES[r['family']] for r in positive) or '无场景')+'。'
 template_note='；'.join(NAMES[r['family']]+'净改善出现在'+str(r['template_coverage']['improved'])+'/'+str(r['template_coverage']['total'])+'个模板' for r in positive)+'。这些是历史材料上的描述性迁移结果，不是新的独立确认实验。'
 floor_note=f"另有{len(floors)}个场景的本次B/D主实验基线已全部正确，干预后仍正确。因此，本轮不能证明对子类型各异的原有错误都有效；尤其不能据此声称已修复这些场景历史A/C提示下的错误。"
 diag_note=f"明确目标证据的历史样本共有{supported_stats['planned']}对，可干预{supported_stats['available']}对；其中正确数从{supported_stats['base_correct']}变为{supported_stats['after_correct']}，新增错误{supported_stats['harm']}个。覆盖不足的样本保留自然结果并在下文单列。"
 reverse_data=[]
 for r in mainrows:
  if r['reverse']['harm']:
   tr=r['reverse']['transitions'];reverse_data.append([NAMES[r['family']],r['reverse']['harm'],tr.get('correct → string_null',0),tr.get('correct → source',0),tr.get('correct → control',0),tr.get('correct → format_failure',0)])
 chunks[2:2]=['## 结果摘要',overview,template_note,floor_note,diag_note,'目前支持的结论是：公司任务中提取的有限状态成分，在部分活动地点样本上也能改变输出，表现出跨场景迁移。保留明确证据回答仍不足以排除“提高缺失信息时的留空倾向”这一解释；尚未证明找到了通用关系绑定机制。多数B/D对照也同时改变了辅助字段的语义与数据类型，这些因素仍未被本实验单独分离。']
 at=chunks.index('## 3. 明确缺失与充分证据诊断')
 chunks[at:at]=['### 反向错误需要区分答案形式与关系误填',table(['场景','反向新增目标错误','变字符串null','误填来源值','误填对照值','结构失败'],reverse_data),'反向干预引起的正确率下降不能全部称为关系混淆。公司新增69个错误中45个是字符串“null”；活动新增20个错误中18个是字符串“null”。这部分变化直接涉及答案形式或选择，而不是把某个实际城市归给了错误关系。它进一步要求我们区分关系信息与最终输出决策。']
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(13,5));labels=[r['family'] for r in mainrows];x=list(range(10))
  for k,offset,color in [('forward',-.24,'#167f88'),('random',0,'#999999'),('full',.24,'#d17d37')]:axes[0].bar([i+offset for i in x],[100*r[k]['net']/r[k]['available'] if r[k]['available'] else 0 for r in mainrows],width=.24,label=k,color=color)
  axes[0].set_ylabel('Target accuracy change (percentage points)');axes[0].axhline(0,color='black',lw=.6);axes[0].legend()
  axes[1].bar([i-.18 for i in x],[r['forward']['repair'] for r in mainrows],width=.36,label='repaired',color='#167f88');axes[1].bar([i+.18 for i in x],[r['forward']['harm'] for r in mainrows],width=.36,label='harmed',color='#c4533a');axes[1].set_ylabel('Count');axes[1].legend()
  for ax in axes:ax.set_xticks(x,labels,rotation=50,ha='right')
  fig.tight_layout();fig.savefig(R/'transfer_summary.png',dpi=160);plt.close(fig);chunks.insert(chunks.index('## 2. 历史十场景主实验')+2,'![迁移概览](transfer_summary.png)')
 except ImportError:pass
 md='\n\n'.join(chunks)+'\n';(R/'report.md').write_text(md,encoding='utf-8')
 def inline(s):
  s=html.escape(s);s=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2" style="max-width:100%">',s);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s);return re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
 body=[]
 for c in chunks:
  if c.startswith('| '):
   out=[]
   for i,line in enumerate(c.splitlines()):
    if i==1:continue
    tag='th' if i==0 else 'td';out.append('<tr>'+''.join(f'<{tag}>'+inline(cell.strip())+f'</{tag}>' for cell in line.strip('|').split('|'))+'</tr>')
   body.append('<div class="scroll"><table>'+''.join(out)+'</table></div>')
  elif c.startswith('#'):
   n=len(c)-len(c.lstrip('#'));body.append(f'<h{n}>'+inline(c[n:].strip())+f'</h{n}>')
  else:body.append('<p>'+inline(c).replace('\n','<br>')+'</p>')
 (R/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>冻结子空间十场景迁移</title><style>body{font:16px/1.8 system-ui;max-width:1500px;margin:30px auto;padding:20px;color:#213547}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}th{background:#eff4f7}.scroll{overflow:auto}a{color:#16729a}</style><body>'+''.join(body)+'</body></html>',encoding='utf-8')
 print(json.dumps(audit));print(json.dumps(mainrows,ensure_ascii=False))
if __name__=='__main__':main()

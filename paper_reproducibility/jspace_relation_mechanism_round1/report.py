"""Summarize locked results without selecting or changing interventions."""
import json, random, statistics, html
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
def read(name):return json.loads((R/name).read_text(encoding='utf-8'))
def rows(name):return [json.loads(s) for s in (R/name).read_text(encoding='utf-8').splitlines() if s.strip()]
def save(name,x):(R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def mean(xs):return sum(xs)/len(xs) if xs else None
def boot(values):
 rng=random.Random(20261015);samples=sorted(mean(rng.choices(values,k=len(values))) for _ in range(2000));return [samples[49],samples[1949]]
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in data])
def pct(x):return '不可用' if x is None else f'{100*x:.1f}%'
def pp(x):return f'{100*x:+.1f}'
def main():
 assert read('status.json')['stage']=='complete','Only finalized runs can be reported'
 natural=rows('natural.jsonl');replay={r['job_id']:r for r in rows('replay.jsonl')};jobs={r['job_id']:r for r in natural};lens=rows('readouts.jsonl');cand=rows('candidates.jsonl');patch=rows('interventions.jsonl');coverage=rows('coverage.jsonl');gate=read('gate_decision.json');selection=read('selection.json')
 assert len(natural)==384 and len(jobs)==384
 assert len(cand)==1152 and all(len([x for x in cand if x['job_id']==j])==3 for j in jobs)
 summary=dict(gate=gate,selection=selection,natural_count=len(natural),pn_available=sum(c['available'] for c in coverage),replay_mismatches=sum(not r['natural_equal'] for r in replay.values()),truncations=sum(r['truncated'] for r in natural),self_checks=sum(r['kind']=='self' for r in patch),behavior=[],interaction={},readout_aggregates=[],intervention={})
 for split in ['development','validation']:
  for evidence in ['absent','supported','swapped']:
   for form in ['weak','direct']:
    for cond in ['B','D']:
     rs=[r for r in natural if (r['split'],r['evidence'],r['form'],r['condition'])==(split,evidence,form,cond)]
     summary['behavior'].append(dict(split=split,evidence=evidence,form=form,condition=cond,n=len(rs),correct=sum(r['correct'] for r in rs),office=sum(r['office_misuse'] for r in rs),null=sum(r['true_null'] for r in rs),invalid=sum(not r['format_valid'] for r in rs)))
  vals=[];bytemplate={}
  for t in sorted({j['template_id'] for j in natural if j['split']==split}):
   rates={}
   for f in ['weak','direct']:
    for c in ['B','D']:
     rates[f,c]=mean([j['office_misuse'] for j in natural if j['template_id']==t and j['evidence']=='absent' and j['form']==f and j['condition']==c])
   val=rates['weak','D']-rates['weak','B']-rates['direct','D']+rates['direct','B'];vals.append(val);bytemplate[t]=val
  summary['interaction'][split]=dict(estimate=mean(vals),ci=boot(vals),templates=bytemplate)
 # Aggregate city A/B differences separately by evidence; swap tracking is HQ minus office.
 for split in ['development','validation']:
  for position in ['P0','PN','P1']:
   for method in ['J-lens','Logit Lens']:
    for layer in sorted({r['layer'] for r in lens if r['method']==method}):
     for form in ['weak','direct']:
      for cond in ['B','D']:
       subset=[r for r in lens if r['position']==position and r['method']==method and r['layer']==layer and (jobs[r['job_id']]['split'],jobs[r['job_id']]['form'],jobs[r['job_id']]['condition'])==(split,form,cond)]
       for ev in ['absent','supported','swapped']:
        use=[r for r in subset if jobs[r['job_id']]['evidence']==ev];diff=[]
        for r in use:
         j=jobs[r['job_id']];first=j['office_city'] if ev=='absent' else j['hq_city'];second=j['city_b'] if ev=='absent' else j['office_city']
         if first in r['scores'] and second in r['scores']:diff.append(r['scores'][first]-r['scores'][second])
        summary['readout_aggregates'].append(dict(split=split,position=position,method=method,layer=layer,form=form,condition=cond,evidence=ev,n=len(diff),mean_difference=mean(diff)))
 if selection['selected']:
  for kind in ['forward','reverse','full','random']:
   rs=[r for r in patch if r['split']=='validation' and r['kind']==kind];stats={}
   for ev in ['absent','supported','swapped']:
    use=[r for r in rs if r['evidence']==ev];stats[ev]=dict(n=len(use),correct=sum(r['correct'] for r in use),baseline_correct=sum(replay[r['job_id']]['correct'] for r in use),null=sum(r['true_null'] for r in use),office=sum(r['office_misuse'] for r in use),transitions=dict(Counter(replay[r['job_id']]['category']+' → '+r['category'] for r in use)))
   summary['intervention'][kind]=stats
  forward=[r for r in patch if r['split']=='validation' and r['kind']=='forward' and r['evidence']=='absent'];template_gains={}
  for t in sorted({r['template_id'] for r in forward}):template_gains[t]=sum(r['true_null']-replay[r['job_id']]['true_null'] for r in forward if r['template_id']==t)/4
  fs=summary['intervention']['forward'];random_stats=summary['intervention']['random'];ng=fs['absent']['correct']-fs['absent']['baseline_correct'];rg=random_stats['absent']['correct']-random_stats['absent']['baseline_correct']
  source_to_null=sum(replay[r['job_id']]['category']=='office' and r['true_null'] for r in forward);format_to_null=sum(not replay[r['job_id']]['format_valid'] and r['true_null'] for r in forward)
  success=len(forward)==16 and all(fs[e]['n']==16 for e in fs) and ng>=4 and sum(v>0 for v in template_gains.values())>=3 and all(fs[e]['correct']>=15 and fs[e]['correct']-fs[e]['baseline_correct']>=-1 for e in ['supported','swapped']) and source_to_null>format_to_null and ng>rg
  summary['selective_repair']=dict(passed=success,null_gain_count=ng,random_gain_count=rg,source_to_null=source_to_null,format_to_null=format_to_null,template_gains=template_gains,ci=boot(list(template_gains.values())))
 summary['swap_tracking']=[]
 lookup={(r['job_id'],r['position'],r['method'],r['layer']):r for r in lens}
 for method in ['J-lens','Logit Lens']:
  for layer in [29,33]:
   for form in ['weak','direct']:
    for cond in ['B','D']:
     tracked=[];changes=[]
     for j in natural:
      if (j['split'],j['evidence'],j['form'],j['condition'])!=('validation','supported',form,cond):continue
      first=lookup.get((j['job_id'],'PN',method,layer));second=lookup.get((j['job_id'].replace('_supported_','_swapped_'),'PN',method,layer))
      if not first or not second:continue
      a,b=j['city_a'],j['city_b']
      if not all(c in first['scores'] and c in second['scores'] for c in [a,b]):continue
      x=first['scores'][a]-first['scores'][b];y=second['scores'][a]-second['scores'][b]
      tracked.append(x<0 and y>0);changes.append(y-x)
     summary['swap_tracking'].append(dict(method=method,layer=layer,form=form,condition=cond,n=len(tracked),both_hq_preferred=sum(tracked),mean_swap_change=mean(changes)))
 save('summary.json',summary)
 chunks=['# 第一轮机制实验：字段要求如何影响事实使用','本报告仅总结本轮新材料，旧实验不加入分母。模型为 Qwen3-8B；384 条自然生成，探索与验证各192条。总部字段始终先输出。', '## 1. 核心结果']
 if not gate['passed']:chunks.append('**探索集未达到预定干预启动门槛，因此本轮完成行为与内部读出观察，没有执行子空间或完整状态干预。** 这不是干预失败，也不能据此判断模型内部不存在相关机制；它说明这批新材料不足以支撑按原计划开展该项因果检验。')
 else:chunks.append('探索集通过预定门槛，完成冻结配置后的验证干预。选择性修复标准：**'+('通过' if summary['selective_repair']['passed'] else '未通过')+'**。子空间为字段要求差异子空间，不能直接称为关系绑定方向。')
 if selection['selected']:
  fs=summary['intervention']['forward'];rs=summary['intervention']['random'];sc=summary['selective_repair']
  chunks.append(f"验证集自然生成的缺失分句 D 条件为7/16来源误填；当前未干预重放为6/16。因此因果比较使用重放基线：4维替换后6/16降到0/16，正确留空从{fs['absent']['baseline_correct']}/16升至{fs['absent']['correct']}/16；同范数随机对照仅升至{rs['absent']['correct']}/16。明确总部与关系交换条件均保持16/16正确。不能将自然与重放之间自行改变的那1条计作干预修复。")
  chunks.append('**没有通过总体成功门槛的原因：改善只出现在 T07、T08 两个模板，为2/4，而预定要求至少3/4。** T05、T06的分句重放本来已经全部正确，没有可修复错误；仍保留原门槛，不在看到结果后改分母。留空增益为37.5个百分点，模板配对描述性95%区间为[0.0, 75.0]个百分点，反映模板数量少且效应集中的不确定性。')
 chunks.append(table(['探索门槛指标','观察值','要求'],[['分句 D−B',pp(gate['weak_D_minus_B'])+' 个百分点','≥12.5'],['直接连接 D−B',pp(gate['direct_D_minus_B'])+' 个百分点','小于分句'],['分句同向模板',str(sum(v>0 for v in gate['template_differences'].values()))+'/4','≥3/4'],['充分证据总部正确率',pct(gate['supported_accuracy']),'≥90%']]))
 chunks+=['## 2. 材料、提示词与划分','8 个新模板 × 4 组实体 × 3 种证据条件 × 2 种关系表达 × B/D 两种要求。T01—T04 为探索，T05—T08 为验证；公司名及城市配对分开，城市词汇集合相同。B 为总部城市＋办公城市，D 为总部城市＋成立年份。条件包括总部未提及、总部与办公室城市不同、交换两城市关系。正文在 B/D 间完全一致。','通用提示保留只依文本、证据不足填 JSON null 的要求；未添加场景专属的防混淆提示。统一单条、BF16、eager attention、贪心、128 token。验证集仅在门槛决定／配置冻结后开始生成。','示例（探索模板，仅用于说明）：\n\n> '+natural[0]['user_prompt'].replace('\n','\n> '),'## 3. 全部行为结果','“来源误填”指将办公室城市填入总部；“正确”含缺失条件留空与充分证据条件正确城市。格式失败保留在分母。']
 for split,label in [('development','探索集'),('validation','验证集')]:
  chunks.append('### '+label)
  data=[[r['evidence'],r['form'],r['condition'],f"{r['correct']}/{r['n']}",f"{r['office']}/{r['n']}",r['invalid']] for r in summary['behavior'] if r['split']==split]
  chunks.append(table(['证据','表达','条件','总部正确','来源误填','格式失败'],data));inter=summary['interaction'][split]
  chunks.append(f"缺失条件配对交互 (D−B)分句−(D−B)直接：{pp(inter['estimate'])} 个百分点；模板配对 bootstrap 描述性95%区间 [{pp(inter['ci'][0])}, {pp(inter['ci'][1])}]。只有4个模板，不能代替独立复现。")
 chunks.append('### 配对输出案例')
 contrasting=[j for j in natural if j['split']=='validation' and j['evidence']=='absent' and j['form']=='weak' and j['condition']=='D' and j['office_misuse'] and jobs[j['job_id'][:-1]+'B']['correct']]
 examples=[]
 for j in contrasting:
  if j['template_id'] not in [x['template_id'] for x in examples]:examples.append(j)
  if len(examples)==2:break
 if not examples:chunks.append('验证集中没有“同一分句材料 B 正确、D 误填办公室”的配对案例；不另选其他标准制造对比。')
 for j in examples:
  group=j['group_id'];cases=[jobs[group+'_'+ev+'_'+form+'_'+cond] for ev,form,cond in [('absent','weak','B'),('absent','weak','D'),('absent','direct','D'),('supported','weak','D'),('swapped','weak','D')]]
  chunks.append(table(['案例','材料','原始输出'],[[x['job_id'],x['material'],x['raw_output'].replace('\n',' ')] for x in cases]))
 chunks+=['## 4. J-lens 与完整候选概率','J-lens 复用既有官方实现与已保存权重，没有重新校准；模型权重、配置、lens 哈希已核验。其原校准材料为英文短文本，不能假定对本任务已经充分校准。零起算层9、13、17、21、25、29、33及最终层35；并列报告 Logit Lens。P0为JSON前，PN为自然总部值前，P1为统一强制前缀。','城市差值不与裸 null 直接作为答案竞争解释。充分证据读数报告“总部城市−办公城市”，正值说明读出偏向真实总部；缺失条件报告“办公室城市−另一个候选城市”，只反映词汇偏向。']
 for method in ['J-lens','Logit Lens']:
  data=[]
  for ev in ['absent','supported','swapped']:
   for form in ['weak','direct']:
    for cond in ['B','D']:
     rs=[r for r in summary['readout_aggregates'] if (r['split'],r['position'],r['method'],r['evidence'],r['form'],r['condition'])==('validation','PN',method,ev,form,cond) and r['layer'] in [29,33]]
     data.append([ev,form,cond]+[('不可用' if not r['n'] else f"{r['mean_difference']:.3f}") for r in rs])
  chunks+=['### 验证集自然 PN：'+method,table(['证据','表达','条件','第29层','第33层'],data)]
 chunks+=['### 同一组城市交换关系后，读数是否跟随总部',table(['方法','层','表达','条件','交换前后均偏向真实总部','城市甲−乙分差的交换增量'],[[r['method'],r['layer'],r['form'],r['condition'],f"{r['both_hq_preferred']}/{r['n']}",'不可用' if r['mean_swap_change'] is None else f"{r['mean_swap_change']:.3f}"] for r in summary['swap_tracking']]),'此处按同一实体配对：交换前城市甲属于办公室、城市乙属于总部，交换后反过来。增量为正说明读数随总部归属方向变化；“两次均偏向真实总部”要求交换前后都选对城市。两项均不能单独证明读数代表完整的关系知识。']
 cdata=[]
 for ev in ['absent','supported','swapped']:
  for form in ['weak','direct']:
   for cond in ['B','D']:
    ids=[j['job_id'] for j in natural if (j['split'],j['evidence'],j['form'],j['condition'])==('validation',ev,form,cond)];correctp=[];win=[]
    for jid in ids:
     cs=[c for c in cand if c['job_id']==jid];correctp.append(next(c['candidate_set_probability'] for c in cs if c['value']==jobs[jid]['hq_city']));win.append(max(cs,key=lambda c:c['logp'])['value']==jobs[jid]['hq_city'])
    cdata.append([ev,form,cond,pct(mean(correctp)),f'{sum(win)}/{len(win)}'])
 chunks+=['### 统一前缀的完整候选比较',table(['证据','表达','条件','正确候选集合内概率均值','正确候选排名第一'],cdata),'使用原模型计算完整的 null、带引号的城市甲／乙及相同逗号分隔符的续写概率。集合内概率不是自然生成概率，也不是 J-lens 完整答案概率。强制前缀会改变计算，应与上面的自然行为分开解释。']
 chunks.append('## 5. 干预与错误转移')
 if not selection['selected']:chunks.append('按预定停止规则未执行干预，未拟合或挑选子空间，因而没有修复成功率、反向干预或随机对照结果。不得用空结果冒充零效应。')
 else:
  chosen=selection['selected'];chunks.append(f"冻结配置：第{chosen['layer']}层、秩{chosen['rank']}。探索集选择性条件通过：{chosen['eligible']}。同事实 B/D 配对，单次替换 PN 残差的投影分量。")
  data=[]
  for kind,stats in summary['intervention'].items():
   for ev,s in stats.items():data.append([kind,ev,s['baseline_correct'],s['correct'],s['null'],s['office'],s['transitions']])
  chunks.append(table(['干预','证据','原正确','干预正确','null','来源误填','错误转移'],data));chunks.append('选择性修复检查：`'+json.dumps(summary['selective_repair'],ensure_ascii=False)+'`')
 chunks+=['## 6. 质量检查与解释边界',f"自然生成384条；PN可定位 {summary['pn_available']}/384；重放与自然续写不同 {summary['replay_mismatches']} 条；自然截断 {summary['truncations']} 条；自替换检查 {summary['self_checks']} 次。干预始终以当前重放为基线。",'本轮读出可以说明关系交换是否改变候选城市偏向，不能单独区分“关系知识已丢失”与“关系仍存但回答选择错误”。有限末端位置干预也不能证明识别了全部处理路径。即便修复成功，供体包含另一抽取要求下的计算结果，仍可能携带答案或语法信息。',
 '在充分证据条件，第29层 J-lens 的16/16配对均在交换前后偏向真实总部；第33层分句 D 为15/16。Logit Lens也呈现相近结果。这说明模型在明确给出总部时可读出随关系变化的信息，**不能直接推广为“缺失总部且误填时，内部仍然完整区分办公室与总部”**。本轮没有直接测量后者所需的关系特异表征。',
 '本轮能推进的判断是：总部值生成前，第33层一个从字段要求差异提取的4维子空间，对部分新模板的输出具有因果影响；同事实供体替换未破坏本轮充分证据回答。完整状态对照也达到相同正确数，因此不能宣称4维方法比完整替换更优，也不能把它唯一解释为关系绑定。自然 PN 的表达/语法差异与答案倾向仍是解释候选。',
 '后续值得优先检验的是：冻结本轮子空间后，在独立材料中确认其适用范围，并进一步区分字段要求、输出语法与答案倾向。此次不追加新材料、不改变门槛或选择新层。',
 '## 7. 文件与复查','[锁定协议](../protocol.json) · [材料与提示词](../data/jobs.json) · [自然输出](natural.jsonl) · [自然重放](replay.jsonl) · [逐层读数](readouts.jsonl) · [完整候选概率](candidates.jsonl) · [门槛决定](gate_decision.json) · [冻结决定](selection.json) · [干预输出](interventions.jsonl) · [机器可读汇总](summary.json) · [环境与溯源](environment.json) · [完成检查](completion_audit.json)']
 # Standard scientific plotting, kept independent of inference and selection.
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
  for row,method in enumerate(['J-lens','Logit Lens']):
   for col,ev in enumerate(['absent','supported','swapped']):
    ax=axes[row,col]
    for form,ls in [('weak','-'),('direct','--')]:
     for cond,color in [('B','#187f8c'),('D','#c35335')]:
      rs=[r for r in summary['readout_aggregates'] if (r['split'],r['position'],r['method'],r['evidence'],r['form'],r['condition'])==('validation','PN',method,ev,form,cond) and r['n']]
      ax.plot([r['layer'] for r in rs],[r['mean_difference'] for r in rs],ls,color=color,label=form+' '+cond)
    ax.axhline(0,color='gray',lw=.7);ax.set_title(method+' / '+ev);ax.set_xlabel('Decoder block (zero-based)');ax.set_ylabel('City logit contrast');ax.legend(fontsize=7)
  fig.tight_layout();fig.savefig(R/'layer_curves.png',dpi=160);plt.close(fig)
  chunks.insert(chunks.index('## 5. 干预与错误转移'),'![验证集自然PN逐层读数](layer_curves.png)\n\n缺失条件为办公室城市减另一个候选城市；两种充分证据条件为总部城市减办公室城市。')
 except ImportError:chunks.append('绘图库不可用；完整逐层数值已保存在 readouts.jsonl 与 summary.json。')
 md='\n\n'.join(chunks)+'\n';(R/'report.md').write_text(md,encoding='utf-8')
 # A small deterministic Markdown renderer for this report's headings/tables/links.
 import re
 def inline(s):
  s=html.escape(s);s=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2" style="max-width:100%">',s);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s);s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s);return s
 parts=[]
 for chunk in chunks:
  if chunk.startswith('| '):
   lines=chunk.splitlines();trs=[]
   for i,line in enumerate(lines):
    if i==1:continue
    tag='th' if i==0 else 'td';trs.append('<tr>'+''.join('<'+tag+'>'+inline(c.strip())+'</'+tag+'>' for c in line.strip('|').split('|'))+'</tr>')
   parts.append('<div class="scroll"><table>'+''.join(trs)+'</table></div>')
  elif chunk.startswith('#'):
   level=len(chunk)-len(chunk.lstrip('#'));parts.append(f'<h{level}>'+inline(chunk[level:].strip())+f'</h{level}>')
  else:parts.append('<p>'+inline(chunk).replace('\n','<br>')+'</p>')
 page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>第一轮机制实验报告</title><style>body{font:16px/1.8 system-ui;max-width:1300px;margin:32px auto;padding:24px;color:#203347}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef4f8}.scroll{overflow:auto}a{color:#16729a}</style><body>'+''.join(parts)+'</body></html>'
 (R/'report.html').write_text(page,encoding='utf-8');print(json.dumps(dict(gate=gate,intervention=summary.get('selective_repair'),rows=len(natural)),ensure_ascii=False))
if __name__=='__main__':main()

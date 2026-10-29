from pathlib import Path
from collections import Counter
import json,random,html,csv
R=Path(__file__).resolve().parent;cfg=json.loads((R/'protocol.json').read_text(encoding='utf-8'));summary={}
def loadlines(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()] if p.exists() else []
def get(p,kind,cond='D'):
 return next((r for r in p['rows'] if r['kind']==kind and r['uid'].endswith('_'+cond)),None)
def ci(values):
 g=random.Random(20261026);n=len(values);bs=sorted(sum(g.choices(values,k=n))/n for _ in range(2000));return [bs[49],bs[1949]]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 核心独立验证扩充实验','', '本报告仅统计新增验证集，旧验证结果不并入。每模型24个新模板×4组新实体×3种证据×2种表达×2种指令，共1152条自然生成；每条件96例。独立措辞模板为24组，生成次数不等于独立样本数。','', '沿用Qwen3.5-2B第22层一维、Qwen3-8B第33层四维的原冻结子空间与随机基。自然行为按严格协议评分；干预按与图8一致的内容评分，同时保留严格结果。所有运行最多128个新token，不自动延长。','']
tablecsv=[]
for key in cfg['models']:
 out=R/'results'/key;ps=loadlines(out/'pairs.jsonl');status=json.loads((out/'status.json').read_text(encoding='utf-8')) if (out/'status.json').exists() else {}
 complete=status.get('stage')=='complete' and len(ps)==576
 sm=dict(status=status,completed_pairs=len(ps),planned_pairs=576,complete=complete)
 lines += [f'## {key}', '',f'进度：{status.get("completed",len(ps))}/576组；已读取记录{len(ps)}组；状态：{status.get("stage","pending")}。','']
 if not complete:
  lines+=['未完成，不作最终结果判断。',''];summary[key]=sm;continue
 assert len({p['pair_id'] for p in ps})==576
 natural={};lines+=['### 自然生成（严格评分）','','|证据|表达|要求|正确/96|正确留空/96|来源误填/96|协议或截断失败/96|','|---|---|---|---:|---:|---:|---:|']
 for ev in ['absent','supported','swapped']:
  for form in ['weak','direct']:
   for cond in ['B','D']:
    rr=[get(p,'natural',cond)['strict'] for p in ps if p['evidence']==ev and p['form']==form];assert len(rr)==96
    stats=dict(n=96,correct=sum(r['correct'] for r in rr),true_null=sum(r['true_null'] and r['correct'] for r in rr),source=sum(r['source_misuse'] for r in rr),failure=sum(not r['format_valid'] for r in rr));natural[f'{ev}_{form}_{cond}']=stats
    lines.append(f'|{ev}|{form}|{cond}|{stats["correct"]}|{stats["true_null"]}|{stats["source"]}|{stats["failure"]}|')
 ts=sorted({p['template_id'] for p in ps});template=[]
 for t in ts:
  item=dict(template=t)
  for form in ['weak','direct']:
   pp=[p for p in ps if p['template_id']==t and p['evidence']=='absent' and p['form']==form]
   item[form+'_gap_pp']=25*sum(int(get(p,'natural','D')['strict']['source_misuse'])-int(get(p,'natural','B')['strict']['source_misuse']) for p in pp)
  item['interaction_pp']=item['weak_gap_pp']-item['direct_gap_pp'];template.append(item)
 behavior=dict(weak_gap_pp=sum(t['weak_gap_pp'] for t in template)/24,direct_gap_pp=sum(t['direct_gap_pp'] for t in template)/24,interaction_pp=sum(t['interaction_pp'] for t in template)/24,interaction_ci=ci([t['interaction_pp'] for t in template]),positive_weak_templates=sum(t['weak_gap_pp']>0 for t in template))
 lines+=['',f'分句D−B：{behavior["weak_gap_pp"]:.1f}个百分点；直接连接：{behavior["direct_gap_pp"]:.1f}个百分点；交互：{behavior["interaction_pp"]:.1f}个百分点，模板配对95%描述性区间[{behavior["interaction_ci"][0]:.1f}, {behavior["interaction_ci"][1]:.1f}]。分句正差模板{behavior["positive_weak_templates"]}/24。','', '### 冻结子空间干预（分句）','','|证据|PN可用/96|未干预正确|低维正确|随机正确|完整正确|反向：B正确→干预正确|','|---|---:|---:|---:|---:|---:|---:|']
 causal={};transitions={};strict={}
 for ev in ['absent','supported','swapped']:
  pp=[p for p in ps if p['form']=='weak' and p['evidence']==ev];assert len(pp)==96
  available=[p for p in pp if p['available']];stats={'planned':96,'available':len(available)}
  for kind in ['replay','forward','random','full']:
   rr=[get(p,kind) or get(p,'replay') or get(p,'natural') for p in pp]
   stats[kind]=sum(r['correct'] for r in rr);strict[f'{ev}_{kind}']=sum(r['strict']['correct'] for r in rr)
  stats['reverse_baseline']=sum((get(p,'replay','B') or get(p,'natural','B'))['correct'] for p in pp)
  stats['reverse']=sum((get(p,'reverse','B') or get(p,'replay','B') or get(p,'natural','B'))['correct'] for p in pp)
  stats['gained']=sum(not get(p,'replay')['correct'] and get(p,'forward')['correct'] for p in available)
  stats['harmed']=sum(get(p,'replay')['correct'] and not get(p,'forward')['correct'] for p in available)
  causal[ev]=stats;transitions[ev]=dict(Counter(get(p,'replay')['category']+' -> '+get(p,'forward')['category'] for p in available))
  lines.append(f'|{ev}|{len(available)}|{stats["replay"]}|{stats["forward"]}|{stats["random"]}|{stats["full"]}|{stats["reverse_baseline"]}→{stats["reverse"]}|')
 for t in template:
  pp=[p for p in ps if p['form']=='weak' and p['evidence']=='absent' and p['template_id']==t['template']]
  t['before']=sum((get(p,'replay') or get(p,'natural'))['correct'] for p in pp);t['after']=sum((get(p,'forward') or get(p,'replay') or get(p,'natural'))['correct'] for p in pp);t['gain_pp']=25*(t['after']-t['before']);tablecsv.append(dict(model=key,**t))
 main=causal['absent'];gain=(main['forward']-main['replay'])/96*100;gainci=ci([t['gain_pp'] for t in template]);positive=sum(t['gain_pp']>0 for t in template)
 source_to_null=transitions['absent'].get('source -> correct',0);format_to_null=sum(n for tr,n in transitions['absent'].items() if tr in ['format_failure -> correct','truncated -> correct'])
 checks=dict(null_gain_at_least25pp=gain>=25,positive_templates_at_least18=positive>=18,supported_retained=causal['supported']['forward']>=90 and causal['supported']['replay']-causal['supported']['forward']<=6,swapped_retained=causal['swapped']['forward']>=90 and causal['swapped']['replay']-causal['swapped']['forward']<=6,source_repairs_exceed_format=source_to_null>format_to_null,beats_random=main['forward']>main['random'],all_pn_available=sum(c['available'] for c in causal.values())==288)
 selfchecks=0
 for p in ps:
  for cond in ['B','D']:
   r=get(p,'self_'+cond,cond)
   if r:assert r['generated_token_ids']==get(p,'replay',cond)['generated_token_ids'];selfchecks+=1
 checks['self_coverage_complete']=selfchecks==576
 checks['all_executed_self_checks_passed']=True
 lines+=['',f'主要留空增益：{gain:.1f}个百分点，模板配对95%描述性区间[{gainci[0]:.1f}, {gainci[1]:.1f}]；改善模板{positive}/24；实际执行的{selfchecks}次自替换均与未干预token完全一致（计划576次）；缺少检查仅表示配对位置不可用，不表示自替换失败。',f'预先固定的扩充选择性标准：{"通过" if all(checks.values()) else "未全部通过"}。', '', '未定位配对保留在96例分母中，不计为改善；表内相应未干预结果采用可用重放，否则采用自然输出。完整可用组不涉及此回退。','', '### 逐模板','','|模板|分句D−B(pp)|直接D−B(pp)|交互(pp)|未干预→低维（缺失，每组4例）|','|---|---:|---:|---:|---|']
 for t in template:lines.append(f'|{t["template"]}|{t["weak_gap_pp"]:.1f}|{t["direct_gap_pp"]:.1f}|{t["interaction_pp"]:.1f}|{t["before"]}→{t["after"]}|')
 lines+=['','### 错误转移（仅可定位配对）','']
 for ev,counts in transitions.items():
  lines += [f'- {ev}：'+ '；'.join(f'{k}: {v}' for k,v in counts.items())]
 sm.update(natural=natural,behavior=behavior,causal=causal,strict_causal_counts=strict,template=template,transitions=transitions,checks=checks,selective_success=all(checks.values()),gain_pp=gain,gain_ci=gainci,self_checks=selfchecks)
 summary[key]=sm
save(R/'results/summary.json',summary);md='\n'.join(lines);(R/'results/report.md').write_text(md,encoding='utf-8')
# Render Markdown tables using standard-library HTML only.
blocks=[];in_table=False
for line in lines:
 if line.startswith('|'):
  if line.startswith('|---'):continue
  if not in_table:blocks.append('<table>');in_table=True
  blocks.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in line.strip('|').split('|'))+'</tr>')
 else:
  if in_table:blocks.append('</table>');in_table=False
  if line.startswith('#'):n=len(line)-len(line.lstrip('#'));blocks.append(f'<h{n}>'+html.escape(line[n:].strip())+f'</h{n}>')
  elif line:blocks.append('<p>'+html.escape(line)+'</p>')
if in_table:blocks.append('</table>')
(R/'results/report.html').write_text('<!doctype html><meta charset="utf-8"><title>扩充独立验证</title><style>body{font:16px/1.65 sans-serif;max-width:1200px;margin:40px auto;color:#34332e}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #ddd;padding:7px}tr:first-child{background:#edf1f4}h2{margin-top:40px}</style>'+''.join(blocks),encoding='utf-8')
if tablecsv:
 with (R/'results/template_results.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(tablecsv[0]));w.writeheader();w.writerows(tablecsv)
print(json.dumps({k:dict(complete=v['complete'],pairs=v['completed_pairs'],gain=v.get('gain_pp'),success=v.get('selective_success')) for k,v in summary.items()},ensure_ascii=False))

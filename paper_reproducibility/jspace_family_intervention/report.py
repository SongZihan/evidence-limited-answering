import json,html,collections,random,math
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'results'
NAMES=dict(company='公司总部',person='人物出生地',event='活动举办地',product='产品生产地',research='研究资助机构',shipment='包裹收货地',film='故事发生地',artifact='文物出土地',book='著作权机构',procurement='合同签订日期')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def load(p):
 if not p.exists():return []
 lines=p.read_text(encoding='utf-8').splitlines();out=[]
 for i,s in enumerate(lines):
  try:out.append(json.loads(s))
  except json.JSONDecodeError:
   if i!=len(lines)-1:raise
 return out
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(str(c).replace('|','／') for c in r)+' |' for r in rows])
def get(p,kind,cond=None):return next((r for r in p['rows'] if r['kind']==kind and (cond is None or r['uid'].endswith('_'+cond) or '_'+cond+'_' in r['uid'])),None)
def stats(ps,kind,strict=False):
 s=dict(planned=len(ps),available=0,before_correct=0,after_correct=0,before_source=0,after_source=0,before_null=0,after_null=0,repair=0,harm=0,transitions={})
 trans=collections.Counter()
 for p in ps:
  r=get(p,kind)
  if r is None:continue
  b=next(x for x in p['rows'] if x['kind']=='replay' and x['uid']==r['uid'])
  if strict:r,b=r['strict'],b['strict']
  s['available']+=1
  for key,name in [('correct','correct'),('source_misuse','source'),('true_null','null')]:s['before_'+name]+=int(b[key]);s['after_'+name]+=int(r[key])
  s['repair']+=int(not b['correct'] and r['correct']);s['harm']+=int(b['correct'] and not r['correct']);trans[b['category']+' → '+r['category']]+=1
 s['net']=s['after_correct']-s['before_correct'];s['null_gain']=s['after_null']-s['before_null'];s['transitions']=dict(trans);return s
def bootstrap(ps,kind):
 groups=collections.defaultdict(list)
 for p in ps:groups[p['template_id']].append(p)
 if not groups:return None
 keys=list(groups);rng=random.Random(20261016);vals=[]
 for _ in range(2000):
  sample=[p for t in rng.choices(keys,k=len(keys)) for p in groups[t]];s=stats(sample,kind)
  # Missing PN retains planned denominator and contributes zero improvement.
  vals.append(s['null_gain']/len(sample))
 vals.sort();return [vals[49],vals[1949]]
def main():
 cfg=read(R/'protocol.json');summary={};sections=['# Gemma Muse GLM 与 Kimi 子空间干预补充实验','状态按各模型单独更新；只有标记 complete 的模型完成全量实验。','本轮沿用固定材料和 B/D 提示，在每个模型的公司探索集上重新构建子空间。候选为两个相对深度层 × 秩1/4；冻结后检验公司验证集和历史十场景。先按原生协议分离最终回答；主评分只去掉一个完整外层代码块，严格评分同时保留。原行为门槛是诊断，不作为本轮方法评估的停机条件。','用户授权位置回退：优先取冒号后、值前的自然 token 边界；若不存在，取包含值起点的 token 之前的边界，可能连同字段尾部或标点重新生成。保留自然思考上下文，不强制答案前缀。它是修改位置规则的补充实验，不与旧版精确 PN 实验混算。','若最终回答没有目标字段，仍保留行为分母并报告无法干预。探索集只使用可定位配对，至少每种证据4对且覆盖4个模板；覆盖不足48对的拟合单独标注，不能当作全部48对拟合。','本报告检验子空间干预，不包含新模型的 J-lens 训练或读出；不得将其描述为跨模型 J-lens 机制复现。']
 for key in cfg['models']:
  d=O/key;s=read(d/'status.json') if (d/'status.json').exists() else {'stage':'pending'};ps=load(d/'pairs.jsonl');sel=read(d/'selection.json') if (d/'selection.json').exists() else {};gate=read(d/'gate.json') if (d/'gate.json').exists() else {}
  sections+=['## '+key,f"状态：{s['stage']}；已记录 {len(ps)}/1176 对材料。",f"固定候选层：{cfg['models'][key]['layers']}；选择结果：`{json.dumps(sel.get('selected'),ensure_ascii=False)}`。",f"原行为门槛：`{json.dumps(gate,ensure_ascii=False)}`。"]
  if s['stage']=='failed':sections.append('运行失败，尚无完整结论。错误详见本模型 status.json。')
  if (d/'short_resume_amendment.json').exists():
   amendment=read(d/'short_resume_amendment.json')
   sections+=['### 用户要求缩短运行后的预算变更',f"保留先前完成的{amendment['retained_pairs']}对；剩余{amendment['remaining_pairs']}对改为每次最多512个新token，不重试。已中断的未落盘案例从原提示重新运行。冻结子空间未重新训练。后续验证集属于短预算条件，不能与前段长预算直接视为同设置。",'以下总表包含两段预算，必须结合本分段表解读。']
   regime_rows=[];regime_stats=[]
   for regime in ['original_long_budget','short_512_no_retry']:
    sub=[p for p in ps if p.get('generation_regime','original_long_budget')==regime];st=stats(sub,'forward');tr=sum(r['truncated'] for p in sub for r in p['rows'])
    regime_stats.append(dict(regime=regime,truncated_outputs=tr,**st));regime_rows.append([regime,len(sub),st['available'],st['repair'],st['harm'],tr])
   sections.append(table(['预算段','配对数','可干预','修复','误伤','截断输出数'],regime_rows));(d/'budget_regime_summary.json').write_text(json.dumps(regime_stats,ensure_ascii=False,indent=2),encoding='utf-8')
  coverage=read(d/'development_coverage.json') if (d/'development_coverage.json').exists() else None
  sections.append(f'探索集可定位覆盖：`{json.dumps(coverage,ensure_ascii=False)}`。')
  boundary_counts=collections.Counter(b.get('mode','unknown') for p in ps for b in p['boundaries'].values())
  reasons=collections.Counter(r.get('unavailable_reason') for p in ps for r in p['rows'] if r['kind']=='natural' and r.get('unavailable_reason'))
  sections.append(f'运行位置计数：`{dict(boundary_counts)}`；不能定位原因：`{dict(reasons)}`。')
  allrows=[r for p in ps for r in p['rows']];dc=load(d/'development_candidates.jsonl');selfrows=[r for r in allrows+dc if r['kind'].startswith('self')]
  for p in ps:
   for r in p['rows']:
    if r['kind'].startswith('self'):assert r['generated_token_ids']==next(b['generated_token_ids'] for b in p['rows'] if b['kind']=='replay' and b['uid']==r['uid'])
  replay_by_uid={r['uid']:r for r in allrows if r['kind']=='replay'}
  for r in dc:
   if r['kind'].startswith('self'):assert r['generated_token_ids']==replay_by_uid[r['uid']]['generated_token_ids']
  audit=dict(recorded_pairs=len(ps),available=sum(p['available'] for p in ps),self_checks=len(selfrows),natural_replay_differences=sum(not r['natural_equal'] for r in allrows if r['kind']=='replay'),truncated=sum(r['truncated'] for r in allrows+dc),generation_calls=sum(len(r['attempts']) for r in allrows+dc),strict_natural_format_failures=sum(not r['strict']['format_valid'] for r in allrows if r['kind']=='natural'),semantic_natural_format_failures=sum(not r['format_valid'] for r in allrows if r['kind']=='natural'))
  if s['stage']=='complete':assert len(ps)==1176 and len({p['pair_id'] for p in ps})==1176
  drift=[]
  for p in ps:
   f=get(p,'forward');r=get(p,'random')
   if f and r:
    assert abs(f['intended_delta_norm']-r['intended_delta_norm'])<max(1e-4,f['intended_delta_norm']*1e-5)
    if f['realized_delta_norm']:drift.append(abs(f['realized_delta_norm']-r['realized_delta_norm'])/f['realized_delta_norm'])
  audit['realized_norm_relative_difference_max']=max(drift) if drift else None
  sections+=['### 历史十场景', '计数以可干预配对为分母；自然输出和不可定位边界的样本仍保留。修复和误伤均相对同前缀的未干预重放。']
  mainrows=[];rows=[]
  for fam in NAMES:
   sub=[p for p in ps if p['dataset']!='round1' and p['family']==fam and p['evidence']=='main'];ss={k:stats(sub,k) for k in ['forward','reverse','full','random']};st=ss['forward'];nat=[get(p,'natural','D') for p in sub]
   mainrows.append(dict(family=fam,natural_D_n=len(nat),natural_D_correct=sum(r['correct'] for r in nat),**ss))
   rows.append([NAMES[fam],len(sub),f"{sum(r['correct'] for r in nat)}/{len(nat)}",st['available'],f"{st['before_source']}→{st['after_source']}",f"{st['before_correct']}→{st['after_correct']}",st['repair'],st['harm'],ss['random']['net'],ss['full']['net'],ss['reverse']['net']])
  sections.append(table(['场景','已记录对数','自然D正确','干预可用','来源误填前→后','正确前→后','修复','误伤','随机净增','完整替换净增','反向净增'],rows))
  positions=[]
  for dataset in ['round1','history']:
   for mode in ['both_exact','any_preceding_token']:
    sub=[p for p in ps if p['available'] and ((p['dataset']=='round1' and p['split']=='validation') if dataset=='round1' else p['dataset']!='round1') and ('any_preceding_token' if any(b.get('mode')=='preceding_token' for b in p['boundaries'].values()) else 'both_exact')==mode]
    st=stats(sub,'forward');positions.append(dict(dataset=dataset,position=mode,**st))
  sections+=['### 精确位置与提前一个 token 的结果',table(['材料','位置类型','已记录可定位对数','实际干预','修复','误伤','正确净增'],[[x['dataset'],x['position'],x['planned'],x['available'],x['repair'],x['harm'],x['net']] for x in positions])]
  val=[p for p in ps if p['dataset']=='round1' and p['split']=='validation' and p['form']=='weak'];v={e:stats([p for p in val if p['evidence']==e],'forward') for e in ['main','supported','swapped']};a=v['main'];rv=stats([p for p in val if p['evidence']=='main'],'random');templates={t:stats([p for p in val if p['evidence']=='main' and p['template_id']==t],'forward') for t in sorted({p['template_id'] for p in val})}
  passed=len(val)==48 and all(x['available']==16 for x in v.values()) and a['null_gain']>=4 and sum(x['null_gain']>0 for x in templates.values())>=3 and all(v[e]['after_correct']>=15 and v[e]['net']>=-1 for e in ['supported','swapped']) and a['null_gain']>rv['null_gain'] and a['transitions'].get('source → correct',0)>a['transitions'].get('format_failure → correct',0)
  sections+=['### 独立公司验证集（分句）',table(['证据','可干预/应有','正确前→后','留空增量','修复','误伤'],[[e,f"{v[e]['available']}/16",f"{v[e]['before_correct']}→{v[e]['after_correct']}",v[e]['null_gain'],v[e]['repair'],v[e]['harm']] for e in v]),f"预设选择性修复标准通过：{passed}；模板留空改善数：{sum(x['null_gain']>0 for x in templates.values())}/4；随机对照留空增量：{rv['null_gain']}。"]
  ci=bootstrap([p for p in val if p['evidence']=='main'],'forward') if len(val)==48 and sel.get('selected') else None
  sections.append(f'主要留空增益的模板配对 bootstrap 95% 区间：{ci}（比例单位，4个模板，有限精度）。')
  diagnostics=[]
  for fam in NAMES:
   for e in ['supported','missing','explicit_missing']:
    sub=[p for p in ps if p['dataset']!='round1' and p['family']==fam and p['evidence']==e]
    if sub:
     st=stats(sub,'forward');diagnostics.append([NAMES[fam],e,len(sub),st['available'],f"{st['before_correct']}→{st['after_correct']}",st['harm'],st['after_null']])
  sections+=['### 有证据与明确缺失对照',table(['场景','证据','记录对数','可干预','正确前→后','误伤','干预后null'],diagnostics),'### 审计',f'`{json.dumps(audit,ensure_ascii=False)}`']
  strata=[]
  for fields in sorted({(p['dataset'],p['family'],p['split'],p['form'],p['evidence'],p['target_order']) for p in ps}):
   sub=[p for p in ps if tuple(p[k] for k in ['dataset','family','split','form','evidence','target_order'])==fields]
   for kind in ['forward','reverse','full','random']:strata.append(dict(zip(['dataset','family','split','form','evidence','order'],fields),kind=kind,semantic=stats(sub,kind),strict=stats(sub,kind,True)))
  pertemplate=[]
  for fields in sorted({(p['dataset'],p['family'],p['template_id'],p['form'],p['evidence'],p['target_order']) for p in ps}):
   sub=[p for p in ps if tuple(p[k] for k in ['dataset','family','template_id','form','evidence','target_order'])==fields];pertemplate.append(dict(zip(['dataset','family','template','form','evidence','order'],fields),**stats(sub,'forward')))
  for name,obj in [('boundary_summary.json',dict(positions=positions,counts=dict(boundary_counts),unavailable_reasons=dict(reasons))),('stratified_summary.json',strata),('template_summary.json',pertemplate),('audit.json',audit),('main_summary.json',mainrows),('validation_summary.json',dict(passed=passed,evidence=v,templates=templates,bootstrap=ci)),('unavailable.json',[p['pair_id'] for p in ps if not p['available']])]:
   (d/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
  summary[key]=dict(status=s,selection=sel,gate=gate,audit=audit,validation_passed=passed,main=mainrows)
 reference=R.parent/'jspace_relation_transfer/results/main_summary.json'
 if reference.exists():
  ref=read(reference);rr=[]
  for p in ref:
   v=p['forward'];rr.append([NAMES[p['family']],v['available'],f"{v['base_source']}→{v['after_source']}",v['repair'],v['harm'],p['random']['net']])
  sections+=['## 已完成的 Qwen3-8B 参照', '来自上一轮冻结第33层、秩4的结果；不计入本轮三个新增模型的运行次数。本轮按相对层深选候选位置，因此不是同一层编号的直接对照。',table(['场景','可干预','来源误填前→后','修复','误伤','随机净增'],rr)]
 sections+=['## 解释边界','原门槛未通过或探索选择不具选择性的模型，结果只能算预先规定的探索性方法评估。历史十场景不是新材料独立验证，拟合只使用公司探索集。多数场景若没有基线错误，正确率保持不能当作修复证据。公司历史充分证据有总部与办公室同城的样本，重点结合本轮异城与关系交换对照。','替换同时改变了供体状态中的语法、回答倾向等信息，有限子空间效果不等于已经识别关系绑定机制。随机对照只有一个固定种子。自然生成与重放可能不同，必须以重放为干预基线，并保留自然输出用于行为分析。','每次干预需要同材料的 B 运行提供供体状态，属于有供体的因果方法测试，不是无需额外运行的部署纠错器。']
 text='\n\n'.join(sections);(O/'report.md').write_text(text,encoding='utf-8');(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 # Render markdown tables and headings without an external runtime dependency.
 parts=[];intable=False
 for line in text.splitlines():
  if line.startswith('|'):
   cells=[s.strip() for s in line.strip('|').split('|')]
   if all(set(s)<=set('-: ') for s in cells):continue
   if not intable:parts.append('<table>');intable=True
   parts.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>');continue
  if intable:parts.append('</table>');intable=False
  if line.startswith('#'):
   level=len(line)-len(line.lstrip('#'));parts.append(f'<h{level}>'+html.escape(line[level:].strip())+f'</h{level}>')
  elif line:parts.append('<p>'+html.escape(line)+'</p>')
 if intable:parts.append('</table>')
 (O/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>多模型子空间干预</title><style>body{font:16px/1.7 system-ui;margin:32px auto;max-width:1400px;padding:24px;color:#182536}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #ccd6df;padding:8px}tr:first-child{background:#e8eff7;font-weight:bold}p{overflow-wrap:anywhere}h2{margin-top:48px}</style>'+''.join(parts),encoding='utf-8')
if __name__=='__main__':main()

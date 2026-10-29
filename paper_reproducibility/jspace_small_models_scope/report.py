import json,html,time
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):
 try:return json.loads((R/p).read_text(encoding='utf-8'))
 except (FileNotFoundError,json.JSONDecodeError):return {}
p=read('protocol.json');lines=['# 小模型十场景复现实验','', '更新时间：'+time.strftime('%Y-%m-%d %H:%M:%S %z'),'','固定历史材料，单模型 3516 条；原始 BF16、贪心解码、128 token 硬上限，不补跑截断。主分析严格 JSON，另列仅移除外层代码围栏的内容分析。未完成项目不计为阴性结果。','', '| 模型 | 状态 | 当前数据集进度 |','|---|---|---|']
for k,c in p.get('models',{}).items():
 s=read(f'results/{k}_status.json');d=read(f'results/{k}_download.json');lines.append(f"| {c['name']} | {s.get('stage',d.get('stage','待开始'))} | {s.get('dataset','')} {s.get('completed','')}/{s.get('total','')} |")
for mode in ['strict','content']:
 lines+=['','## '+('严格格式' if mode=='strict' else '去除外层代码围栏后的辅助分析'),'','| 模型 | 场景 | 分类/主要差值 | 充分证据正确率 | 主样本协议失败率 |','|---|---|---|---|---|']
 for k,c in p.get('models',{}).items():
  s=read(f'results/{k}_company_{mode}_summary.json')
  if s:
   v=s['validation']['continuation'];a=v['primary'];lines.append(f"| {c['name']} | 公司（验证集） | D−B={a['estimate']:.1%}；同向模板 {v.get('positive_templates','?')}；local gate={v.get('local_gate')} | 见逐条数据 | 见逐条数据 |")
  for ds in ['four','five']:
   for f,x in read(f'results/{k}_{ds}_{mode}_summary.json').get('families',{}).items():
    quality=x['support_pass'] and x['main_protocol_failure_rate']<=.05
    label={'full':'完整复现','partial':'部分复现','not_replicated':'未达复现门槛'}[x['label']] if quality else '证据不足：任务/格式质量未达标'
    lines.append(f"| {c['name']} | {f} | {label} | {x['supported_correct_rate']:.1%} | {x['main_protocol_failure_rate']:.1%} |")
lines+=['','## 严格评分下的核心对比','','下表全部使用 D−B；新九场景筛选还保留预定的其他字段对比，不能把筛选所得最大对比误当成预注册 D−B。百分点为绝对差。','','| 模型 | 场景 | 分句 D−B | 直接连接 D−B | 表达交互 |','|---|---|---|---|---|']
for k,c in p.get('models',{}).items():
 s=read(f'results/{k}_company_strict_summary.json')
 if s:
  contrasts=s['validation']['contrasts'];vals=[next(x['estimate'] for x in contrasts if x['form']==f and x['order']=='average' and x['contrast']=='D-B') for f in ['weak','direct','weak-minus-direct']]
  lines.append('| '+c['name']+' | 公司验证集 | '+' | '.join(f'{v*100:+.1f} pp' for v in vals)+' |')
 for ds in ['four','five']:
  for f,x in read(f'results/{k}_{ds}_strict_summary.json').get('families',{}).items():
   a=next(z for z in x['contrasts'] if z['contrast']=='D-B');lines.append('| '+c['name']+' | '+f+' | '+' | '.join(f"{a[z]['mean']*100:+.1f} pp" for z in ['weak','direct','interaction'])+' |')
lines+=['','本轮属于历史测试材料上的跨模型范围探索，不能证明普遍性；同一系列多个尺度不算独立系列。公司判据与新九场景判据不同，不汇总成统一成功率。截断、错误及失败模型保留；技术失败与行为阴性分开。完整细分、逐模板差值及区间保存在各模型 summary.json，原始提示、输出和评分保存在 JSONL。']
text='\n'.join(lines)+'\n'
for name,body in [('report.md',text),('report.html','<!doctype html><meta charset="utf-8"><title>小模型十场景实验</title><style>body{max-width:1200px;margin:40px auto;font:16px/1.7 sans-serif}pre{white-space:pre-wrap}</style><h1>小模型十场景复现实验</h1><pre>'+html.escape(text)+'</pre>')]:
 t=R/'results'/name;t.with_suffix('.tmp').write_text(body,encoding='utf-8');t.with_suffix('.tmp').replace(t)

import json,html,hashlib,importlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in (ROOT/p).read_text(encoding='utf-8').splitlines()]
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def esc(x):return html.escape(str(x))
def table(headers,rs):return '<div style="overflow:auto"><table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rs)+'</table></div>'
def pp(c):return f"{100*c['estimate']:+.1f} [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}]"
def cp(c):return f"{100*c['mean']:+.1f} [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}]"
names={'8b':'Qwen3-8B','2b':'Qwen3.5-2B','27b':'Qwen3.5-27B'};labels={'full':'完整筛选通过','partial':'部分复现','not_replicated':'未达到标准'}
assert read('results/pipeline_complete.json')['complete']
for p,h in read('data/lock.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
allrows={};summaries={};audits={};paths={}
for model in names:
 for ds in ['company','four','five']:
  path=f'data/8b_{ds}.jsonl' if model=='8b' else 'data/2b_five.jsonl' if model=='2b' and ds=='five' else f'results/{model}_{ds}.jsonl'
  paths[model,ds]=path;rows=lines(path);jobs=read(f'data/{ds}_jobs.json');assert len(rows)==len(jobs) and len({r['job_id'] for r in rows})==len(jobs)
  score=importlib.import_module(ds+'_scoring').score;analyze=importlib.import_module(ds+'_analyze').analyze;cfg=read(f'data/{ds}_config.json')
  for r,j in zip(rows,jobs):
   assert r['job_id']==j['job_id'] and r['user_prompt']==j['user_prompt'] and r['material']==j['material'] and r['gold']==j['gold']
   issues=r.get('issues',r.get('format_issues',[]))
   for k,v in score(j,r['raw_output'],'truncated' in issues).items():assert r[k]==v,(model,ds,r['job_id'],k)
  allrows[model,ds]=rows
  if ds=='company':summary={split:analyze([r for r in rows if r['split']==split],cfg) for split in ['development','validation']}
  else:summary=analyze(rows,cfg)
  summaries[model,ds]=summary
  save(f'results/{model}_{ds}_audited_summary.json',summary)
  audits[f'{model}_{ds}']=dict(n=len(rows),exact_job_order_prompt_gold_match=True,rescore_matches=True,structural_failures=sum(not r['format_valid'] for r in rows),protocol_failures=sum(not r['protocol_valid'] for r in rows),string_null=sum(r['string_null'] for r in rows),sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),source=path)
save('results/completion_audit.json',dict(all_models_scenarios_complete=True,per_model_n=3516,new_generations=5592,reused_small_five=1440,outputs=audits))
def mainrows(model,ds,family):
 return [r for r in allrows[model,ds] if r['family']==family and (r['relation_form'] in ['weak','direct','label'] if ds=='company' else r['evidence']=='main')]
def source(r,ds):return bool(r['office_city_misuse']) if ds=='company' else r['source_misuse']
def form(r,ds):return r['relation_form'] if ds=='company' else r['form']
p=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>十场景三模型对比</title><style>body{font:17px/1.75 system-ui,sans-serif;background:#f4f7fa;color:#213547;max-width:1330px;margin:35px auto;padding:0 24px}section{background:white;border-radius:12px;padding:24px;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}th{background:#edf2f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:12px;font-size:14px}a{color:#176b99}</style><h1>十场景三模型对比</h1><p>补齐 Qwen3.5-2B 公司＋前四场景，再用 Qwen3.5-27B 跑完整十场景。每模型可比记录 3516 条；本次新增 5592 次生成，复用已完成的 2B 五场景 1440 条及历史 8B 记录。</p>']
p.append('<section><h2>结果要点</h2><p><b>原公司场景的“字段组合×关系表达”交互没有形成稳定的跨模型复现。</b>验证交互在 Qwen3-8B 为 +47.2 个百分点，在 Qwen3.5-2B 为 +5.6（区间跨零），在 Qwen3.5-27B 为零。2B 虽然仍有 +25 个百分点的分句 D−B 差异、通过旧公司行为门槛，但这不等于表达交互已复现。</p><p>另外九场景中，活动在 8B 和 2B 上属于部分复现；27B 未达到任何场景的复现门槛。27B 来源误填仅有活动的 3 次，但另有 11 次裸 null 违反 JSON 对象格式（公司3次、活动8次），所以不能称为所有任务全部正确。</p><p>小模型在多个关系上更容易误填，但错误数量增加本身不构成核心现象复现。2B 科研成果的来源误填虽为零，仍有 34 次误用数据提供机构；下方总表只列来源误填，应结合逐条件表中的对照值和编码错误阅读。没有观察到误填的条件存在地板效应，不能据此断言内部过程或真实总体效应不存在。</p></section>')
p.append('<section><h2>范围与可比性</h2><p>公司采用关系归属实验的 924 条正式任务，包括探索／验证、五种表达和字段顺序；不重复 70 条技术先导。前四场景采用已删除针对性提醒的 1152 条任务，后五场景为 1440 条。此前同一关系的重复难度版本与内部替换实验不作为新场景重复计算。</p><p>全部模型使用相同各组材料、用户提示、金标准和任务顺序。两款 Qwen3.5 均使用服务器现有环境，BF16、eager、禁用思考、贪心、最多128个新token；公司批次12，其余16。原生聊天模板因模型而异。2B 先前五场景的结束标记分离已核验，本次在生成时直接按原生结束token终止。</p><p>公司设计与其他九场景不同，不能合并分母；同代2B与27B也并非参数量单因素实验。所有场景和历史结果均保留，正式研究仍需新的未见材料。</p></section>')
overview=[];scenario_list=[('company','company','公司总部')]
for ds in ['four','five']:
 scenario_list += [(ds,f,v['name']) for f,v in read(f'data/{ds}_config.json')['families'].items()]
for ds,f,title in scenario_list:
 row=[title]
 for model in names:
  rr=mainrows(model,ds,f);row.append(f"{sum(source(r,ds) for r in rr)}/{len(rr)}")
 overview.append(row)
p.append('<section><h2>来源值误填总数</h2>'+table(['场景']+list(names.values()),overview)+'<p>公司分母756，包含两数据划分、三种无证据表达和双字段顺序；其他场景分母256。此表仅描述来源值误填，不是核心现象复现判定；不包括对照值误填、字符串 null 和结构失败。</p></section>')
classification=[]
for ds in ['four','five']:
 for f,v in read(f'data/{ds}_config.json')['families'].items():classification.append([v['name']]+[labels[summaries[m,ds]['families'][f]['label']] for m in names])
p.append('<section><h2>九个跨场景先导：沿用原筛选标准</h2>'+table(['场景']+list(names.values()),classification)+'<p>字段效应绝对值≥12.5pp且至少5/8模板同号；同一比较的表达交互也需满足两项；来源占有效非空错误≥60%，充分证据至少15/16正确且B、D各至少7/8，协议失败≤5%。有合格字段效应但无合格交互为部分复现；不因更换模型调阈值。</p></section>')
p.append('<section><h2>公司场景：原验证标准与交互</h2>')
company=[]
for m in names:
 dev=summaries[m,'company']['development'];val=summaries[m,'company']['validation'];primary=val['continuation']['primary'];inter=next(x for x in val['contrasts'] if x['form']=='weak-minus-direct' and x['contrast']=='D-B' and x['order']=='average');gate=val['continuation']['local_gate'] and dev['continuation']['primary']['estimate']>0
 company.append([names[m],pp(primary),len(val['continuation']['positive_templates']),pp(inter),'通过' if gate else '未通过'])
p.append(table(['模型','验证分句 D−B pp [CI]','D−B正向模板/6','分句−直接交互 pp [CI]','原公司行为门槛'],company)+'<p>公司门槛要求验证D−B≥10pp、区间下界&gt;0、至少4/6模板正向、探索集同向且协议失败差异&lt;5pp。此处保留原规则，不能与九场景更严格的先导标签直接等同；这些模板已用于历史研究，并非新的独立确认集。</p></section>')
for ds,f,title in scenario_list:
 p.append('<section><h2>'+title+'：逐条件结果</h2>')
 for m in names:
  rr=[r for r in allrows[m,ds] if r['family']==f];p.append('<h3>'+names[m]+'</h3>');cells=[]
  if ds=='company':
   for split in ['development','validation']:
    for formname in ['weak','direct','label','missing','supported']:
     for cond in 'ABCD':
      xs=[r for r in rr if r['split']==split and r['relation_form']==formname and r['condition']==cond]
      cells.append([split,formname,cond,len(xs),sum(bool(r['office_city_misuse']) for r in xs),sum(bool(r['launch_city_misuse']) for r in xs),sum(r['target_category']=='json_null' for r in xs),sum(r['string_null'] for r in xs),sum(not r['format_valid'] for r in xs),sum(r['field_correct']['总部城市'] for r in xs)])
   p.append(table(['划分','表达','条件','n','办公误用','发布会误用','null','字符串null','结构失败','目标正确'],cells))
  else:
   v=summaries[m,ds]['families'][f]
   p.append(table(['表达','条件','n','来源误用','对照误用','其他非空','null','字符串null','结构失败','目标正确'],[[x['form'],x['condition'],x['n'],x['source'],x['control'],x['other'],x['null'],x['string_null'],x['invalid'],x['target_correct']] for x in v['cells']]))
   p.append(table(['比较','分句 pp [CI]','直接 pp [CI]','交互 pp [CI]','交互同号模板/8','完整标准'],[[x['contrast'],cp(x['weak']),cp(x['direct']),cp(x['interaction']),x['interaction']['same_sign_templates'],'通过' if x['full'] else '未通过'] for x in v['contrasts']]))
 p.append('</section>')
p.append('<section><h2>核验与文件</h2><p>所有记录完整、任务ID唯一、与原任务顺序及提示词逐条一致，使用原评分器重算一致。仅保留适量的2000次模板配对bootstrap；没有注意力头扫描或追加机制实验。全零经验区间不证明总体风险为零。</p>')
quality=[]
for m in names:
 diag=[]
 for ds in ['company','four','five']:
  for r in allrows[m,ds]:
   if (r['relation_form'] in ['missing','supported'] if ds=='company' else r['evidence']!='main'):diag.append(bool(r['field_correct']['总部城市']) if ds=='company' else r['target_correct'])
 quality.append([names[m],f'{sum(diag)}/{len(diag)}',sum(audits[f'{m}_{ds}']['structural_failures'] for ds in ['company','four','five']),sum(audits[f'{m}_{ds}']['string_null'] for ds in ['company','four','five'])])
p.append(table(['模型','明确缺失／充分证据诊断目标正确','结构失败','字符串 null'],quality))
for m in names:
 for ds in ['company','four','five']:
  sourcepath='../'+paths[m,ds] if paths[m,ds].startswith('data/') else Path(paths[m,ds]).name
  p.append('<p>'+names[m]+' / '+ds+'：<a href="'+sourcepath+'">逐条输出</a> · <a href="'+f'{m}_{ds}_audited_summary.json'+'">完整统计与模板差值</a></p>')
p.append('<p><a href="completion_audit.json">完成核验</a> · <a href="../protocol.json">运行前协议</a> · <a href="2b_environment.json">2B环境</a> · <a href="27b_environment.json">27B环境</a></p></section></html>')
(ROOT/'results/report.html').write_text(''.join(p),encoding='utf-8')
save('results/comparison.json',dict(source_counts=overview,screening=classification,company=company,audits=audits))
print(json.dumps(dict(source_counts=overview,screening=classification,company=company),ensure_ascii=False,indent=2))

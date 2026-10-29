import json, hashlib, html, importlib, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'jspace_all_models'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def table(headers,rows):
 return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'
def pp(x,key='mean'): return f"{100*x[key]:+.1f} [{100*x['ci95'][0]:+.1f}, {100*x['ci95'][1]:+.1f}]"

assert read(ROOT/'results/gemma12b_complete.json')['complete']
for p,h in read(ROOT/'data/lock.json').items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
allrows={}; summaries={}; audits={}
for ds in ['company','four','five']:
 path=ROOT/f'results/gemma12b_{ds}.jsonl'
 rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
 jobs=read(ROOT/f'data/{ds}_jobs.json');cfg=read(ROOT/f'data/{ds}_config.json')
 assert (ROOT/f'data/{ds}_jobs.json').read_bytes()==(OLD/f'data/{ds}_jobs.json').read_bytes()
 assert len(rows)==len(jobs)==len({r['job_id'] for r in rows})
 scorer=importlib.import_module(ds+'_scoring').score
 for r,j in zip(rows,jobs):
  assert all(r[k]==v for k,v in j.items()),j['job_id']
  issues=r.get('issues',r.get('format_issues',[]))
  assert all(r[k]==v for k,v in scorer(j,r['raw_output'],'truncated' in issues).items())
 strict_rows=rows
 rows=[];fences=0
 for r,j in zip(strict_rows,jobs):
  match=re.fullmatch(r'\s*```(?:json)?\s*\n([\s\S]*?)\n```\s*',r['raw_output'])
  body=match.group(1) if match else r['raw_output']
  fences+=bool(match)
  rows.append({**r,**scorer(j,body,'truncated' in r.get('issues',r.get('format_issues',[]))),'original_raw_output':r['raw_output'],'raw_output':body,'outer_fence_removed':bool(match)})
 (ROOT/f'results/gemma12b_{ds}_fence_view.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
 analyze=importlib.import_module(ds+'_analyze').analyze
 summary={s:analyze([r for r in rows if r['split']==s],cfg) for s in ['development','validation']} if ds=='company' else analyze(rows,cfg)
 allrows[ds]=rows;summaries[ds]=summary
 save(ROOT/f'results/gemma12b_{ds}_audited_summary.json',summary)
 audits[ds]=dict(n=len(rows),original_jobs_exact_match=True,rescore_matches=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),strict_structural_failures=sum(not r['format_valid'] for r in strict_rows),outer_fences_removed=fences,structural_failures=sum(not r['format_valid'] for r in rows),string_null=sum(r['string_null'] for r in rows))

old=read(OLD/'results/comparison.json');overview=[];screens=[];details=[];diagnostics=[]
labels={'full':'完整筛选通过','partial':'部分复现','not_replicated':'未达到标准'}
company=[r for r in allrows['company'] if r['relation_form'] in ['weak','direct','label']]
overview.append(old['source_counts'][0]+[f"{sum(bool(r['office_city_misuse']) for r in company)}/{len(company)}"])
index=1
for ds in ['four','five']:
 for family,v in read(ROOT/f'data/{ds}_config.json')['families'].items():
  ss=summaries[ds]['families'][family];cells=ss['cells'];main=[x for x in cells if x['form'] in ['weak','direct']]
  overview.append(old['source_counts'][index]+[f"{sum(x['source'] for x in main)}/{sum(x['n'] for x in main)}"])
  screens.append([v['name'],labels[ss['label']]])
  details.append('<section><h2>'+v['name']+'</h2>'+table(['表达','条件','n','来源误用','对照误用','其他非空','真null','字符串null','结构失败','目标正确','辅助字段正确/n'],[[x['form'],x['condition'],x['n'],x['source'],x['control'],x['other'],x['null'],x['string_null'],x['invalid'],x['target_correct'],str(x['auxiliary_correct'])+'/'+str(x['n_aux'])] for x in cells]))
  details.append(table(['比较','分句差值 pp [CI]','直接差值 pp [CI]','表达交互 pp [CI]','交互同号模板/8','完整筛选'],[[x['contrast'],pp(x['weak']),pp(x['direct']),pp(x['interaction']),x['interaction']['same_sign_templates'],x['full']] for x in ss['contrasts']])+'</section>')
  index+=1
for ds,rr in allrows.items():
 diag=[r for r in rr if (r['relation_form'] in ['missing','supported'] if ds=='company' else r['evidence']!='main')]
 correct=sum(r['field_correct']['总部城市'] if ds=='company' else r['target_correct'] for r in diag)
 diagnostics.append([ds,f'{correct}/{len(diag)}',audits[ds]['strict_structural_failures'],audits[ds]['outer_fences_removed'],audits[ds]['structural_failures'],audits[ds]['string_null']])
val=summaries['company']['validation'];dev=summaries['company']['development']
primary=val['continuation']['primary'];inter=next(x for x in val['contrasts'] if x['form']=='weak-minus-direct' and x['contrast']=='D-B' and x['order']=='average')
gate=val['continuation']['local_gate'] and dev['continuation']['primary']['estimate']>0
company_stats=dict(primary=primary,interaction=inter,original_gate=gate,positive_templates=val['continuation']['positive_templates'])
save(ROOT/'results/completion_audit.json',audits)
save(ROOT/'results/comparison.json',dict(source_counts=overview,screening=screens,company=company_stats,diagnostics=diagnostics))
p=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Gemma 4 12B：十场景跨系列实验</title><style>body{font:16px/1.8 system-ui;background:#f3f6fa;color:#203040;max-width:1250px;margin:30px auto;padding:20px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eaf0f6}a{color:#126d9b}</style><h1>Gemma 4 12B：十场景跨系列实验</h1>']
p.append('<section><h2>必须区分：严格协议与辅助内容分析</h2><p>首批60条回答均在JSON外添加Markdown代码块，原始评分器因此判定格式失败。本轮保持提示词、原始回答与严格评分不变，另建立辅助视图：只在整个回答匹配一个完整的外层代码块时，删除该外壳；不修改内部文本、不修补JSON、不提取嵌入解释的片段。</p><p><b>下方Gemma关系错误和筛选标签均基于这个辅助视图，不能作为原严格协议通过的结论。</b>该辅助分析是在看到输出格式后追加，原始严格汇总和完整输出单独保留。外层代码块是模型生成的文本，不是可忽略的原生传输标记。四模型表用于描述字段内容，不是严格协议成绩的同口径比较。</p></section>')
p.append('<section><h2>本轮结果</h2><p>3516条全部完成。全部回答带外层代码块，严格协议下均为格式失败。辅助内容分析中，只有活动出现场景预设的来源值误填：31/256（12.1%）；其他九场景未观察到来源值误填。公司验证集分句D−B及表达交互均为0；新增九场景均未达到原筛选门槛。</p><p>活动的C−B在分句/直接条件分别为12.5/15.6个百分点，但只有3/8和2/8模板同向，达不到至少5/8的要求。其表达交互为−3.1个百分点。因此有局部差异，尚不能概括为稳定的跨模板规律。</p><p>辅助视图下，456条明确缺失或充分证据诊断的目标字段均正确；主实验仍有9条辅助字段类型错误（公司3条、科研6条，将字符串输出为数组）。数字附加单位的答案也保留原评分，不做额外归一化。因此不能将低来源误填率理解为整个抽取任务全部正确。</p></section>')
p.append('<section><h2>实验范围</h2><p>服务器已有的 gemma-4-12B-it，十场景3516次新生成。用户提示、正文、金标准、任务顺序与此前三模型实验完全一致；前四场景采用已删除针对性关系提醒的版本。BF16、eager、贪心、128新token上限，原生聊天模板 enable_thinking=False，按模型原生结束token停止。保留原评分与筛选标准。公司批次12，其余16。</p><p>公司主实验756条，其他九场景各256条；另有456条诊断。公司包含不同字段顺序与三种关系表达，不能与九场景合并分母。模型系列、训练、结构与参数量同时变化，不能归因于单一因素。全部模板已在历史研究中使用，本轮为跨模型复测。</p></section>')
p.append('<section><h2>来源值误填：四模型对照</h2>'+table(['场景','Qwen3-8B','Qwen3.5-2B','Qwen3.5-27B','Gemma 4 12B'],overview)+'<p>这里只计算来源值被误填到目标字段；不包含其他值误填或格式错误。零来源误填不等于所有输出正确。</p></section>')
p.append('<section><h2>公司验证集</h2>'+table(['分句D−B pp [CI]','表达交互 pp [CI]','正向模板/6','原公司门槛'],[[pp(primary,'estimate'),pp(inter,'estimate'),len(val['continuation']['positive_templates']),'通过' if gate else '未通过']])+'<p>表达交互为 (D−B)分句−(D−B)直接。原公司门槛与新增九场景筛选不同，不能将通过公司门槛直接当成完整交互复现。</p></section>')
p.append('<section><h2>九场景筛选</h2>'+table(['场景','Gemma 4 12B'],screens)+'<p>字段效应绝对值≥12.5pp、至少5/8模板同向；同一对比的表达交互也需达到这两项。来源错误占比≥60%，充分证据至少15/16正确且B/D各至少7/8，协议失败≤5%。只满足字段效应及质量要求属于部分复现。沿用2000次模板配对bootstrap，不新增统计扫描；未达标准不证明总体效应不存在。</p></section>')
p.extend(details)
p.append('<section><h2>诊断与输出质量</h2>'+table(['数据组','辅助视图诊断正确','原始结构失败','去除外壳数','去外壳后结构失败','字符串null'],diagnostics))
for ds in allrows:
 p.append(f'<p>{ds}：<a href="gemma12b_{ds}.jsonl">原始输出与严格评分</a> · <a href="gemma12b_{ds}_fence_view.jsonl">辅助内容视图</a> · <a href="gemma12b_{ds}_audited_summary.json">辅助视图逐条件及模板统计</a></p>')
 if ds=='company':p.append('<p>原始严格汇总：<a href="gemma12b_company_development_summary.json">探索集</a> · <a href="gemma12b_company_validation_summary.json">验证集</a></p>')
 else:p.append(f'<p><a href="gemma12b_{ds}_summary.json">原始严格汇总</a></p>')
p.append('<p><a href="completion_audit.json">完整性核验</a> · <a href="gemma12b_environment.json">模型与环境</a> · <a href="../protocol.json">运行协议</a> · <a href="../../jspace_all_models/results/report.html">此前三模型报告</a></p></section></html>')
(ROOT/'results/report.html').write_text(''.join(p),encoding='utf-8')
print(json.dumps(dict(source_counts=overview,screening=screens,company=company_stats,diagnostics=diagnostics),ensure_ascii=False,indent=2))

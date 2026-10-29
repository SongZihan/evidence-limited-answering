import json,re,html,hashlib,importlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def table(h,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</table></div>'
def effect(c,k='mean'):return f"{100*c[k]:+.1f} [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}]"
plan=read(ROOT/'protocol.json');names={'muse30b':'Muse Glimmer 30B','kimi16b':'Kimi-VL 16B-A3B','glm9b':'GLM-4.1V-9B-Thinking'}
labels={'full':'完整筛选通过','partial':'部分复现','not_replicated':'未达到标准'}
audits={};summaries={};allrows={};quality=[];company=[];classification=[]
for rel,h in read(ROOT/'data/lock.json').items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
for key in names:
 assert read(ROOT/f'results/{key}_complete.json')['complete'],key
 for ds in plan['datasets']:
  suffix='_native' if key=='kimi16b' else ''
  path=ROOT/f'results/{key}_{ds}{suffix}.jsonl';raw=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
  jobs=read(ROOT/f'data/{ds}_jobs.json');cfg=read(ROOT/f'data/{ds}_config.json')
  assert len(raw)==len(jobs)==len({r['job_id'] for r in raw})
  assert (ROOT/f'data/{ds}_jobs.json').read_bytes()==(ROOT.parent/f'jspace_all_models/data/{ds}_jobs.json').read_bytes()
  score=importlib.import_module(ds+'_scoring').score;analyze=importlib.import_module(ds+'_analyze').analyze
  rows=[];fences=0
  for r,j in zip(raw,jobs):
   assert all(r[k]==v for k,v in j.items()),(key,ds,j['job_id'])
   assert all(r[k]==v for k,v in score(j,r['raw_output'],r['truncated_generation']).items())
   m=re.fullmatch(r'\s*```(?:json)?\s*\n([\s\S]*?)\n```\s*',r['raw_output'])
   body=m.group(1) if m else r['raw_output'];fences+=bool(m)
   rows.append({**r,**score(j,body,r['truncated_generation']),'strict_raw_output':r['raw_output'],'raw_output':body,'outer_fence_removed':bool(m)})
  summary={s:analyze([r for r in rows if r['split']==s],cfg) for s in ['development','validation']} if ds=='company' else analyze(rows,cfg)
  save(ROOT/f'results/{key}_{ds}_content_summary.json',summary)
  (ROOT/f'results/{key}_{ds}_content.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
  summaries[key,ds]=summary;allrows[key,ds]=rows
  audit=dict(n=len(raw),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),exact_jobs_prompts_gold=True,rescore_identical=True,strict_invalid=sum(not r['format_valid'] for r in raw),outer_fences=fences,content_invalid=sum(not r['format_valid'] for r in rows),truncated=sum(r['truncated_generation'] for r in raw),no_final=sum(not r['final_answer_found'] for r in raw),string_null=sum(r['string_null'] for r in rows))
  audits[key+'_'+ds]=audit
  diag=[r for r in rows if (r['relation_form'] in ['missing','supported'] if ds=='company' else r['evidence']!='main')]
  correct=sum(r['field_correct']['总部城市'] if ds=='company' else r['target_correct'] for r in diag)
  quality.append([names[key],ds,len(raw),audit['strict_invalid'],fences,audit['content_invalid'],audit['string_null'],audit['truncated'],audit['no_final'],f'{correct}/{len(diag)}'])
 val=summaries[key,'company']['validation'];dev=summaries[key,'company']['development']
 primary=val['continuation']['primary'];inter=next(x for x in val['contrasts'] if x['form']=='weak-minus-direct' and x['contrast']=='D-B' and x['order']=='average')
 company.append([names[key],effect(primary,'estimate'),len(val['continuation']['positive_templates']),effect(inter,'estimate'),'通过' if val['continuation']['local_gate'] and dev['continuation']['primary']['estimate']>0 else '未通过'])

old=read(ROOT.parent/'jspace_gemma12b/results/comparison.json');overview=[]
scenarios=[('company','company','公司总部')]
for ds in ['four','five']:scenarios.extend((ds,f,v['name']) for f,v in read(ROOT/f'data/{ds}_config.json')['families'].items())
for i,(ds,f,title) in enumerate(scenarios):
 row=list(old['source_counts'][i]);classes=[title]
 for key in names:
  rs=[r for r in allrows[key,ds] if r['family']==f and (r['relation_form'] in ['weak','direct','label'] if ds=='company' else r['evidence']=='main')]
  source=sum(bool(r['office_city_misuse']) if ds=='company' else r['source_misuse'] for r in rs)
  row.append(f'{source}/{len(rs)}')
  if ds!='company':classes.append(labels[summaries[key,ds]['families'][f]['label']])
 overview.append(row)
 if ds!='company':classification.append(classes)
save(ROOT/'results/completion_audit.json',audits)
save(ROOT/'results/comparison.json',dict(source_counts=overview,screening=classification,company=company,quality=quality))
p=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>新增三系列：十场景实验</title><style>body{font:16px/1.8 system-ui;color:#203040;background:#f1f5f9;max-width:1450px;margin:30px auto;padding:20px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}th{background:#e9eff5}a{color:#126a97}</style><h1>新增三系列：十场景实验</h1>']
if (ROOT/'results/repair_manifest.json').exists():
 repair=read(ROOT/'results/repair_manifest.json')
 p.append('<section><h2>截断案例补跑与替换</h2><p>根据用户要求，对首轮全部'+str(repair['total'])+'条截断案例从相同原始提示重新生成，首先提高上限至8192，仍截断则提高至16384；BF16、eager、贪心、权重及原生模板不变，补跑批次最多8条、双卡分配。批量变化可能带来数值差异，逐条记录新生成是否与原2048-token前缀一致。未截断案例保持原记录不变。下方统计已使用补跑替换结果。</p><p>补跑后仍截断：'+str(repair['remaining_truncated'])+'条，仍计为失败，未视作正确留空。预算是针对首轮截断案例自适应增加，不能当作所有模型同预算比较。保留首轮快照供核验。</p><p><a href="repair_manifest.json">逐条替换清单</a> · <a href="before_truncation_repair/report.html">首轮报告快照</a> · <a href="before_truncation_repair/">首轮原始文件</a></p></section>')
elif (ROOT/'results/repair_status.json').exists():
 p.append('<section><h2>首轮结果：截断补跑尚未完成</h2><p>本页当前为首轮结果。已按用户要求安排截断案例补跑，完成后替换并重算；请勿将本页视为补跑后的最终统计。</p><p><a href="progress.html">查看补跑进度</a></p></section>')
p.append('<section><h2>范围与评分口径</h2><p>Muse Glimmer 30B、Kimi-VL 16B-A3B、GLM-4.1V-9B-Thinking，每模型3516条，共10548次正式生成。全部用户提示、材料、金标准与任务顺序沿用此前实验。每模型另有8条技术检查，不计入正式结果。</p><p>原生聊天模板不同：Muse使用low推理强度，GLM保留思考模式，两者总生成上限2048token；Kimi为128。均BF16、eager、贪心。原生推理/最终回答协议与最终答案内容分开保存；缺少最终答案或截断单独记录。跨系列对比同时改变训练、结构、参数规模、推理模式和预算，不能归因于单一因素。</p><p><b>下表来源值误填与筛选使用辅助内容口径：仅去除完整单个外层Markdown代码块，内部答案不修改。</b>原最终回答的严格JSON评分保留在原始输出与strict_summary中，格式质量另列。因此辅助筛选通过不等于原严格协议通过。历史Gemma同样使用去外壳口径；其他历史模型使用其既有评分。</p><p>公司主实验756条、其余场景各256条；公司设计含标签表达、双字段顺序及原探索/验证划分，不能与九场景合并分母。只沿用原2000次模板bootstrap与筛选阈值，无新增机制扫描。模板已用于历史实验，属于跨模型复测。</p></section>')
p.append('<section><h2>Kimi原生结束标记修正</h2><p>Kimi的3516条原始生成均以原生特殊token im_end（163586）加EOS（163585）结束。初始解析只截去了EOS，导致评分器误将im_end当正文，产生全量格式失败。已逐条用原tokenizer验证，仅分离这一个末尾原生标记，内部答案不变。原始生成与初始评分均保留，修正后的严格评分存于native文件。这与删除模型生成的Markdown外壳不同。</p><p><a href="kimi_transport_audit.json">3516条结束标记核验</a></p></section>')
p.append('<section><h2>七模型来源值误填对照</h2>'+table(['场景','Qwen3-8B','Qwen3.5-2B','Qwen3.5-27B','Gemma 4 12B']+list(names.values()),overview)+'<p>来源值误填为零不代表全对；还需检查对照值误填、其他值、格式、截断及辅助字段。大量格式失败的模型不能按零来源误填解释为正确。</p></section>')
p.append('<section><h2>公司原验证集：辅助内容分析</h2>'+table(['模型','分句D−B pp [CI]','正向模板/6','表达交互 pp [CI]','原公司门槛'],company)+'<p>表达交互=(D−B)分句−(D−B)直接。公司原门槛不等于新增九场景完整交互筛选。</p></section>')
p.append('<section><h2>九场景辅助筛选</h2>'+table(['场景']+list(names.values()),classification)+'<p>字段差异绝对值≥12.5pp，至少5/8模板同向；同一对比的表达交互也需同时满足。来源错误占有效非空错误≥60%，支持对照至少15/16且B、D各7/8，协议失败≤5%。只有合格字段效应、无合格表达交互为部分复现。这里协议判断基于辅助内容视图；严格输出格式质量见下表。</p></section>')
p.append('<section><h2>输出质量与诊断</h2>'+table(['模型','数据组','n','严格结构失败','去外壳数','内容结构失败','字符串null','截断','无最终答案','辅助视图诊断正确'],quality)+'</section>')
for ds,f,title in scenarios:
 p.append('<section><h2>'+title+'：逐条件结果</h2>')
 for key in names:
  p.append('<h3>'+names[key]+'</h3>')
  if ds=='company':
   cells=[]
   for split in ['development','validation']:
    for form in ['weak','direct','label','missing','supported']:
     for cond in 'ABCD':
      rr=[r for r in allrows[key,ds] if r['split']==split and r['relation_form']==form and r['condition']==cond]
      cells.append([split,form,cond,len(rr),sum(bool(r['office_city_misuse']) for r in rr),sum(bool(r['launch_city_misuse']) for r in rr),sum(r['target_category']=='json_null' for r in rr),sum(r['string_null'] for r in rr),sum(not r['format_valid'] for r in rr),sum(r['field_correct']['总部城市'] for r in rr)])
   p.append(table(['划分','表达','条件','n','来源误用','对照误用','null','字符串null','结构失败','目标正确'],cells))
  else:
   ss=summaries[key,ds]['families'][f]
   p.append(table(['表达','条件','n','来源误用','对照误用','其他非空','null','字符串null','结构失败','目标正确','辅助字段正确/n'],[[x['form'],x['condition'],x['n'],x['source'],x['control'],x['other'],x['null'],x['string_null'],x['invalid'],x['target_correct'],str(x['auxiliary_correct'])+'/'+str(x['n_aux'])] for x in ss['cells']]))
   p.append(table(['比较','分句 pp [CI]','直接 pp [CI]','交互 pp [CI]','交互同号模板/8','完整筛选'],[[x['contrast'],effect(x['weak']),effect(x['direct']),effect(x['interaction']),x['interaction']['same_sign_templates'],x['full']] for x in ss['contrasts']]))
 p.append('</section>')
p.append('<section><h2>原始文件与核验</h2><p>全部正式记录数、任务ID唯一性、原任务提示词与金标准逐条一致、评分重算一致均已检查。保留所有场景。全零经验区间不能证明总体风险为零。</p>')
for key in names:
 for ds in plan['datasets']:
  suffix='_native' if key=='kimi16b' else ''
  p.append(f'<p>{names[key]} / {ds}：<a href="{key}_{ds}{suffix}.jsonl">最终回答严格评分</a> · <a href="{key}_{ds}{suffix}_strict_summary.json">严格汇总</a> · <a href="{key}_{ds}_content.jsonl">辅助视图</a> · <a href="{key}_{ds}_content_summary.json">辅助统计</a> · <a href="{key}_{ds}.jsonl">原始记录</a></p>')
 p.append(f'<p><a href="{key}_environment.json">{names[key]}环境与权重哈希</a> · <a href="{key}_smoke.json">技术检查</a></p>')
p.append('<p><a href="completion_audit.json">完整性核验</a> · <a href="../protocol.json">协议</a></p></section></html>')
(ROOT/'results/report.html').write_text(''.join(p),encoding='utf-8')
print(json.dumps(dict(source_counts=overview,screening=classification,company=company,quality=quality),ensure_ascii=False,indent=2))

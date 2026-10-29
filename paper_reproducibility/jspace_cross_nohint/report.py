import json,hashlib,html,random,statistics
from pathlib import Path
from analyze import analyze
from scoring import score
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def lines(p):return [json.loads(x) for x in (ROOT/p).read_text(encoding='utf-8').splitlines()]
cfg=read('protocol.json');new=lines('results/outputs.jsonl');old=lines('data/with_hint_outputs.jsonl');jobs=read('data/jobs.json')
assert len(new)==len(old)==len(jobs)==1152
assert [r['job_id'] for r in old]==[r['job_id'] for r in new]
assert len({r['job_id'] for r in new})==1152
for p,h in read('data/lock.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
env=read('results/environment.json');oe=read('data/with_hint_environment.json')
for k in ['model_revision','precision','batch_size','seed','chat_template','torch','transformers']:assert env[k]==oe[k],k
changes=[]
for a,b in zip(old,new):
 hint=cfg['paired_rerun']['only_prompt_change'][a['family']]
 assert a['user_prompt'].replace(hint,'')==b['user_prompt']
 assert a['prompt_text'].replace(hint,'')==b['prompt_text']
 for k in ['material','fields','gold','condition','form','source_value','control_value','supported_value']:assert a[k]==b[k],k
 for k,v in score(b,b['raw_output'],'truncated' in b['issues']).items():assert b[k]==v
 if a['raw_output']!=b['raw_output']:
  changes.append(dict(job_id=b['job_id'],family=b['family'],evidence=b['evidence'],old=a['raw_output'],new=b['raw_output'],old_category=a['target_category'],new_category=b['target_category'],target_changed=a['target_value']!=b['target_value']))
s=analyze(new,cfg);os=analyze(old,cfg);save('results/summary.json',s);save('results/changed_outputs.json',changes)
rng=random.Random(cfg['seed']);samples=[[rng.randrange(8) for _ in range(8)] for _ in range(2000)]
paired={}
for f in cfg['families']:
 pairs=[(a,b) for a,b in zip(old,new) if b['family']==f and b['evidence']=='main'];templates=sorted({b['template_id'] for a,b in pairs})
 means=[statistics.mean(int(b['source_misuse'])-int(a['source_misuse']) for a,b in pairs if b['template_id']==t) for t in templates]
 boot=sorted(statistics.mean(means[i] for i in ix) for ix in samples)
 paired[f]=dict(old_source=sum(a['source_misuse'] for a,b in pairs),new_source=sum(b['source_misuse'] for a,b in pairs),n=len(pairs),delta=statistics.mean(means),ci95=[boot[49],boot[1949]],gained=sum(not a['source_misuse'] and b['source_misuse'] for a,b in pairs),lost=sum(a['source_misuse'] and not b['source_misuse'] for a,b in pairs),old_invalid=sum(not a['format_valid'] for a,b in pairs),new_invalid=sum(not b['format_valid'] for a,b in pairs),old_label=os['families'][f]['label'],new_label=s['families'][f]['label'])
audit=dict(n=1152,paired_job_order_matches=True,only_hint_deleted=True,materials_fields_gold_unchanged=True,model_settings_match=True,scoring_rules_unchanged=True,locked_hashes_match=True,changed_raw_outputs=len(changes),changed_target_values=sum(c['target_changed'] for c in changes),invalid=sum(not r['format_valid'] for r in new),string_null=sum(r['string_null'] for r in new),outputs_sha256=hashlib.sha256((ROOT/'results/outputs.jsonl').read_bytes()).hexdigest())
save('results/paired_comparison.json',dict(audit=audit,families=paired));save('results/completion_audit.json',audit)
def esc(x):return html.escape(str(x))
def table(head,rs):return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rs)+'</table>'
labels={'full':'完整复现筛选标准通过','partial':'部分复现','not_replicated':'未达到复现标准'}
forms={'weak':'分句','direct':'直接连接','missing':'分句＋明确缺失','supported':'分句＋不同目标值证据'}
def pp(x):return f"{100*x['mean']:+.1f} [{100*x['ci95'][0]:+.1f}, {100*x['ci95'][1]:+.1f}]"
p=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>删除针对性提醒：配对复跑</title><style>body{font:17px/1.75 system-ui,sans-serif;background:#f4f7fa;color:#213547;max-width:1200px;margin:35px auto;padding:0 24px}section{background:white;border-radius:12px;padding:24px;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}th{background:#edf2f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:12px;font-size:14px}a{color:#176b99}</style><h1>删除针对性提醒：四场景配对复跑</h1><p>1024 次主实验＋128 次诊断；旧结果完整保留。只删除每场景一句关系不能替代的提醒，其他提示内容、材料、金标准、顺序与模型设置不变。</p>']
p.append('<section><h2>旧结果与新结果</h2>'+table(['场景','有提醒：来源误填','删除提醒：来源误填','配对变化 pp [95% CI]','旧判定 → 新判定'],[[cfg['families'][f]['name'],f"{x['old_source']}/{x['n']}",f"{x['new_source']}/{x['n']}",f"{100*x['delta']:+.1f} [{100*x['ci95'][0]:+.1f}, {100*x['ci95'][1]:+.1f}]",labels[x['old_label']]+' → '+labels[x['new_label']]] for f,x in paired.items()])+'<p>本次达到完整筛选标准的场景数：'+str(s['n_full'])+'/4；'+('达到' if s['advance_to_formal'] else '未达到')+'“至少两个新场景”的正式扩充门槛。配对差值是删除提醒减去保留提醒，主指标仅计有效结构中误用来源值；格式失败另列。</p></section>')
p.append('<section><h2>实际删掉的文字</h2>'+table(['场景','仅删除这句话'],[[cfg['families'][f]['name'],h] for f,h in cfg['paired_rerun']['only_prompt_change'].items()])+'<p>通用指令仍保留“不得利用外部知识、常识推测或其他字段的值补全缺失关系”，因为它与旧公司实验一致，用户要求删除的是额外的针对性提醒。</p></section>')
p.append('<section><h2>结果如何解读</h2><p>删除针对性提醒后，人物、商品、科研成果三个场景各 256 次均正确留空；活动场景仍为部分复现。活动的来源误填从 90/256 变为 81/256，配对减少 3.5 个百分点，模板区间约为 [−9.4, +2.3]，不能据此认定提醒具有稳定的增错或纠错作用。</p><p>此次复跑不支持“那句提醒压住了另外三个场景的来源误填”这一解释，但只涉及本轮材料和模型。128 次明确缺失／明确支持诊断全部正确。旧科研成果的 2 次数组格式错误消失，新活动场景有 2 次字符串 "null"，单列而不计为来源误填。</p></section>')
for f,v in s['families'].items():
 p.append('<section><h2>'+cfg['families'][f]['name']+'：'+labels[v['label']]+'</h2>')
 oc={(c['form'],c['condition']):c for c in os['families'][f]['cells']}
 p.append(table(['表达','条件','旧→新来源误填','新对照值误填','新其他非空','新真实 null','新字符串 null','新结构失败'],[[forms[c['form']],c['condition'],f"{oc[c['form'],c['condition']]['source']} → {c['source']} / {c['n']}",c['control'],c['other'],c['null'],c['string_null'],c['invalid']] for c in v['cells'] if c['form'] in ['weak','direct']]))
 p.append(table(['比较','新分句 pp [CI]','新直接 pp [CI]','新交互 pp [CI]','交互同号模板','完整标准'],[[c['contrast'],pp(c['weak']),pp(c['direct']),pp(c['interaction']),str(c['interaction']['same_sign_templates'])+'/8','通过' if c['full'] else '未通过'] for c in v['contrasts']]))
 p.append(table(['诊断','条件','旧→新目标正确','新真实 null','新结构失败'],[[forms[c['form']],c['condition'],f"{oc[c['form'],c['condition']]['target_correct']} → {c['target_correct']} / {c['n']}",c['null'],c['invalid']] for c in v['cells'] if c['form'] in ['missing','supported']]))
 p.append('<details><summary>所有变化的原始输出</summary>')
 for c in changes:
  if c['family']==f:p.append('<p>'+esc(c['job_id'])+'</p><pre>有提醒：'+esc(c['old'])+'\n无提醒：'+esc(c['new'])+'</pre>')
 p.append('</details></section>')
p.append('<section><h2>规则与核验</h2><p>沿用原规则：字段效应绝对值至少 12.5 个百分点且至少 5/8 模板同号；同一比较的表达交互也需达到这两个条件；来源值占有效非空错误至少 60%，充分证据至少 15/16 正确且 B、D 各至少 7/8，协议失败率不超过 5%。未调阈值寻找成功场景。</p><p>全部 1152 个 job ID 及生成顺序一致，逐条检查只有提醒被删除，所有材料、字段和金标准不变，checkpoint、模型版本与生成配置一致。原始输出变化 '+str(audit['changed_raw_outputs'])+' 条，其中目标值变化 '+str(audit['changed_target_values'])+' 条。新结果结构失败 '+str(audit['invalid'])+' 条，字符串 null '+str(audit['string_null'])+' 条。</p><p>仅采用 2000 次模板配对 bootstrap，没有增加种子扫描。全零样本的区间退化不代表总体风险为零。这是旧材料的提示敏感性复跑，不是全新验证集；删除句子同时改变输入长度，结果不能定位某种内部机制。</p><p><a href="paired_comparison.json">配对变化汇总</a> · <a href="summary.json">完整统计及逐模板差值</a> · <a href="outputs.jsonl">新逐条输出</a> · <a href="changed_outputs.json">所有变化输出</a> · <a href="../data/jobs.json">新完整提示词</a> · <a href="../protocol.json">锁定协议</a> · <a href="../../jspace_cross_scenario/results/report.html">旧报告</a></p></section></html>')
(ROOT/'results/report.html').write_text(''.join(p),encoding='utf-8')
print(json.dumps(dict(audit=audit,paired=paired,n_full=s['n_full']),ensure_ascii=False,indent=2))

import json,hashlib,html,sys
from pathlib import Path
from analyze import analyze
from scoring import score
ROOT=Path(__file__).resolve().parent
def read(root,p):return json.loads((root/p).read_text(encoding='utf-8'))
def save(root,p,x):(root/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def esc(x):return html.escape(str(x))
def table(head,rs):return '<div style="overflow-x:auto"><table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rs)+'</table></div>'
labels={'full':'完整筛选标准通过','partial':'部分复现','not_replicated':'未达到复现标准'}
forms={'weak':'分句','direct':'直接','missing':'明确缺失','supported':'明确支持不同值'}
def pp(x):return f"{100*x['mean']:+.1f} [{100*x['ci95'][0]:+.1f}, {100*x['ci95'][1]:+.1f}]"
def audit(root):
 cfg=read(root,'protocol.json');jobs=read(root,'data/jobs.json');answer_path=root/'results/answer_outputs.jsonl'
 output_path=answer_path if answer_path.exists() else root/'results/outputs.jsonl'
 rows=[json.loads(x) for x in output_path.read_text(encoding='utf-8').splitlines()]
 assert len(rows)==1440 and len({r['job_id'] for r in rows})==1440 and {r['job_id'] for r in rows}=={j['job_id'] for j in jobs}
 for p,h in read(root,'data/lock.json').items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,(root.name,p)
 for r in rows:
  for k,v in score(r,r['raw_output'],'truncated' in r['issues']).items():assert r[k]==v
  assert '不能互相替代' not in r['prompt_text'] and '不能替代' not in r['prompt_text']
  assert r['fields'][0]==r['target_field']
 s=analyze(rows,cfg);save(root,'results/answer_summary.json' if answer_path.exists() else 'results/summary.json',s)
 a=dict(n=1440,main=1280,diagnostic=160,unique_ids=True,locked_hashes_match=True,all_rescored_identically=True,no_specific_relation_warning=True,invalid=sum(not r['format_valid'] for r in rows),string_null=sum(r['string_null'] for r in rows),truncated=sum('truncated' in r['issues'] for r in rows),diagnostic_correct=sum(r['target_correct'] for r in rows if r['evidence']!='main'),outputs_sha256=hashlib.sha256((root/'results/outputs.jsonl').read_bytes()).hexdigest())
 a['scored_output_file']=output_path.name
 if answer_path.exists():
  ta=read(root,'results/transport_audit.json');assert ta['normalized_sha256']==hashlib.sha256(answer_path.read_bytes()).hexdigest();assert ta['raw_sha256']==a['outputs_sha256'];a['transport_normalization']=ta
 save(root,'results/completion_audit.json',a);return cfg,rows,s,a
models=[('Qwen3-8B',ROOT)]
small=ROOT.parent/'jspace_five_small'
if (small/'results/status.json').exists() and read(small,'results/status.json')['stage']=='complete':models.append(('Qwen3.5-2B',small))
packs={name:audit(root) for name,root in models};cfg=packs['Qwen3-8B'][0]
if len(models)==2:
 assert (ROOT/'data/jobs.json').read_bytes()==(small/'data/jobs.json').read_bytes()
 r0=packs['Qwen3-8B'][1];r1=packs['Qwen3.5-2B'][1]
 assert [r['job_id'] for r in r0]==[r['job_id'] for r in r1]
 for a,b in zip(r0,r1):assert a['user_prompt']==b['user_prompt'] and a['gold']==b['gold']
p=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>五个新场景与小模型复测</title><style>body{font:17px/1.75 system-ui,sans-serif;background:#f4f7fa;color:#213547;max-width:1260px;margin:35px auto;padding:0 24px}section{background:white;border-radius:12px;padding:24px;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}th{background:#edf2f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:12px;font-size:14px}a{color:#176b99}</style><h1>五个新场景与小模型复测</h1><p>每模型：1280 次主实验＋160 次诊断；5 场景×8 模板×4 组实体×2 表达×4 字段条件。原生聊天，禁用思考，目标字段先输出，BF16，贪心生成。</p>']
p.append('<section><h2>场景与字段</h2>'+table(['场景','目标字段','来源字段','同类型对照','不同类型对照'],[[v['name']]+v['fields'] for v in cfg['families'].values()])+'<p>A 只抽目标；B 加抽来源；C 加抽同类型对照；D 加抽不同类型对照。四条件同材料同定义，只改变抽取要求。所有新提示均不包含针对性“不能互相替代”提醒；仍保留与旧公司场景相同的通用指令。</p></section>')
top=[]
for name,root in models:
 c,rs,s,a=packs[name]
 for f,v in s['families'].items():
  main=[r for r in rs if r['family']==f and r['evidence']=='main'];top.append([name,c['families'][f]['name'],f"{sum(r['source_misuse'] for r in main)}/256",sum(r['control_misuse'] for r in main),sum(r['other_string'] for r in main),sum(r['string_null'] for r in main),sum(not r['format_valid'] for r in main),labels[v['label']],','.join(v['full_contrasts']) or '无'])
p.append('<section><h2>总体结果（所有场景均保留）</h2>'+table(['模型','场景','来源误填','对照误填','其他非空','字符串 null','结构失败','筛选判定','完整比较'],top))
for name,root in models:
 c,rs,s,a=packs[name];p.append('<p>'+name+'：完整标准通过 '+str(s['n_full'])+'/5；诊断目标正确 '+str(a['diagnostic_correct'])+'/160；结构失败 '+str(a['invalid'])+'，截断 '+str(a['truncated'])+'。</p>')
if len(models)==2:
 p.append('<p><b>8B 只出现少量来源误填；2B 的影视、图书和采购错误明显更多，但均未达到预设跨模板字段效应与表达交互门槛。</b>这不是“所有场景毫无变化”：例如 2B 图书直接表达的 C−A 来源误填差值为 +31.25 个百分点，影视为 +21.875 个百分点，但未满足至少 5/8 模板同号，不能升级为稳定复现。</p><p>2B 影视有 106 次来源值误填及 59 次首映城市误填；图书有 63 次出版机构误填及 29 次印刷机构误填。误填并非只沿预先指定的来源关系发生。图书主实验另有 42 次字符串 null，协议失败率 16.4%，也超过预设 5% 质量门槛。影视与图书可保留为后续范围探索候选，不能仅凭错误增多进入机制实验。</p><p>两模型充分证据诊断均为 80/80 正确。明确缺失诊断：8B 为 80/80，2B 为 78/80；2B 的两个失败都是图书场景将 null 写成字符串。2B 全部 54 次字符串 null 单列，未计作来源误填。</p>')
p.append('<p>“完整”指达到沿用的探索性筛选标准，不等于完成确认性复现。“未达到”也不等于证明总体效应为零。一次筛查多个场景和比较，正式研究需要全新材料，不能以筛选成功的数据冒充验证集。</p></section>')
for name,root in models:
 c,rs,s,a=packs[name]
 for f,v in s['families'].items():
  p.append('<section><h2>'+esc(name+' · '+c['families'][f]['name'])+'</h2>')
  p.append(table(['表达','条件','来源误填','对照误填','其他非空','真实 null','字符串 null','结构失败','辅助字段正确'],[[forms[x['form']],x['condition'],f"{x['source']}/{x['n']}",x['control'],x['other'],x['null'],x['string_null'],x['invalid'],f"{x['auxiliary_correct']}/{x['n_aux']}" if x['n_aux'] else '—'] for x in v['cells'] if x['form'] in ['weak','direct']]))
  p.append(table(['比较','分句差值 pp [95% CI]','直接差值 pp [95% CI]','交互 pp [95% CI]','交互同号模板','完整标准'],[[x['contrast'],pp(x['weak']),pp(x['direct']),pp(x['interaction']),str(x['interaction']['same_sign_templates'])+'/8','通过' if x['full'] else '未通过'] for x in v['contrasts']]))
  p.append(table(['诊断','条件','目标正确','真实 null','字符串 null','结构失败'],[[forms[x['form']],x['condition'],f"{x['target_correct']}/{x['n']}",x['null'],x['string_null'],x['invalid']] for x in v['cells'] if x['form'] in ['missing','supported']]))
  p.append('<details><summary>8 个模板第一组材料与全部条件原始输出</summary>')
  for t in range(1,9):
   gid=f'{f}_t{t:02d}_e1';p.append('<h3>'+gid+'</h3>')
   for form in ['weak','direct']:
    rr=sorted([r for r in rs if r['group_id']==gid and r['form']==form],key=lambda r:r['condition'])
    p.append('<p>'+forms[form]+'</p><pre>'+esc(rr[0]['material'])+'</pre>'+table(['条件','输出'],[[r['condition'],r['raw_output']] for r in rr]))
  p.append('</details></section>')
p.append('<section><h2>预设规则与解释边界</h2><p>沿用前轮标准：至少一种表达的字段组合差值绝对值≥12.5 个百分点、至少 5/8 模板同号；同一比较的表达交互也需满足这两项，才可能完整通过。来源值占有效非空错误≥60%；充分证据对照至少 15/16 正确，B、D 各至少 7/8；主实验协议失败率≤5%。只有合格字段效应而无合格交互归为部分复现。</p><p>只做 2000 次配对模板 bootstrap，不扫描种子，不额外做机制实验。全零经验区间 [0,0] 不说明总体风险严格为零。日期严格按 YYYY-MM-DD 评分，异常日期写法保留在其他非空输出中。五种关系和对应模板在模型输出前锁定。</p><p>Qwen3.5-2B 与 Qwen3-8B 不仅参数规模不同，模型代际、结构、原生聊天模板和 transformers 版本也不同；不能把比较差异单独归因为模型大小。两模型用户提示、金标准、样本及批次顺序、贪心与禁用思考设置一致；没有修改现有虚拟环境。</p></section>')
p.append('<section><h2>可核验文件</h2>')
p.append('<p>小模型的 1440 条原始输出都包含相同原生结束标记 &lt;|im_end|&gt;，随后是换行及 &lt;|endoftext|&gt;。原解析器将该标记算入 JSON 正文而产生误判。现按 tokenizer 验证的特殊 token 248046 截取助手正文，检查尾部仅包含结束标记和空白，不修复任何答案。原始输出、正文输出及 <a href="../../jspace_five_small/results/transport_audit.json">分离核验</a> 均保留；没有重新生成或按结果筛选样本。</p>')
for name,root in models:
 rel='' if root==ROOT else '../../jspace_five_small/results/'
 summary='answer_summary.json' if root==small else 'summary.json'
 p.append('<p>'+name+'：<a href="'+rel+'outputs.jsonl">原始输出</a> · <a href="'+rel+summary+'">逐模板统计</a> · <a href="'+rel+'completion_audit.json">完成核验</a> · <a href="'+rel+'environment.json">运行环境</a></p>')
 if root==small:p.append('<p><a href="'+rel+'answer_outputs.jsonl">小模型助手正文与逐条评分</a></p>')
p.append('<p><a href="../data/jobs.json">共同任务与提示词</a> · <a href="../protocol.json">锁定协议</a></p></section></html>')
(ROOT/'results/report.html').write_text(''.join(p),encoding='utf-8')
save(ROOT,'results/model_comparison.json',{name:dict(audit=packs[name][3],families={f:dict(label=v['label'],full=v['full_contrasts'],field=v['field_effect_contrasts']) for f,v in packs[name][2]['families'].items()}) for name,root in models})
print(json.dumps(read(ROOT,'results/model_comparison.json'),ensure_ascii=False,indent=2))

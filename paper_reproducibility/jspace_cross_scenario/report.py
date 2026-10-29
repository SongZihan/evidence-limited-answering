import json,html,hashlib
from pathlib import Path
from analyze import analyze
from scoring import score
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def save(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
cfg=read('protocol.json');rows=[json.loads(x) for x in (ROOT/'results/outputs.jsonl').read_text(encoding='utf-8').splitlines()];jobs=read('data/jobs.json')
assert len(rows)==1152 and {r['job_id'] for r in rows}=={j['job_id'] for j in jobs} and len({r['job_id'] for r in rows})==1152
for p,h in read('data/lock.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
for r in rows:
 for k,v in score(r,r['raw_output'],'truncated' in r['issues']).items():assert r[k]==v,(r['job_id'],k)
s=analyze(rows,cfg);save('results/summary.json',s)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
english={'person':'Person: work city → birth city','event':'Event: organizer city → current venue','product':'Product: seller city → production city','research':'Research: affiliation → funder'}
for ax,(family,info) in zip(axes.flat,s['families'].items()):
 for offset,form,color,label in [(-.18,'weak','#c27648','Split relation'),(.18,'direct','#287d97','Direct relation')]:
  cc=[next(x for x in info['cells'] if x['form']==form and x['condition']==c) for c in 'ABCD']
  bars=ax.bar([i+offset for i in range(4)],[100*x['source']/x['n'] for x in cc],width=.34,color=color,label=label)
  for bar,x in zip(bars,cc):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{x['source']}/{x['n']}",ha='center',va='bottom',fontsize=8)
 ax.set_title(english[family],fontsize=11);ax.set_xticks(range(4),list('ABCD'));ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0,0].legend(frameon=False,fontsize=9,loc='upper right')
for ax in axes[:,0]:ax.set_ylabel('Source-value misattribution (%)')
fig.suptitle('Cross-scenario scope pilot | all templates retained',fontsize=14)
fig.text(.5,.015,'A: target only   B: +source   C: +same-type control   D: +different-type control\nEach bar: 8 templates × 4 entity groups. Descriptive counts; paired intervals are in the report.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.075,1,.95]);fig.savefig(ROOT/'results/scenario_rates.png',dpi=160);plt.close(fig)
audit=dict(n_rows=len(rows),source_code_and_data_hashes_match=True,all_outputs_rescored_identically=True,unique_job_ids=True,counts_match_plan=True,outputs_sha256=hashlib.sha256((ROOT/'results/outputs.jsonl').read_bytes()).hexdigest(),invalid=sum(not r['format_valid'] for r in rows),string_null=sum(r['string_null'] for r in rows),protocol_failures=sum(not r['protocol_valid'] for r in rows))
save('results/completion_audit.json',audit)
esc=lambda x:html.escape(str(x))
def table(head,rs):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rs)+'</tbody></table>'
def pp(x):return f"{100*x['mean']:+.1f} [{100*x['ci95'][0]:+.1f}, {100*x['ci95'][1]:+.1f}]"
labels=dict(full='达到完整复现筛选标准',partial='部分复现：字段组合效应',not_replicated='未达到核心现象复现标准')
forms=dict(weak='分句表达',direct='直接连接',missing='分句＋明确缺失',supported='分句＋明确支持不同值')
p=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>跨场景关系归属先导</title><style>body{font:17px/1.75 system-ui,sans-serif;color:#203040;background:#f5f7fa;max-width:1250px;margin:36px auto;padding:0 24px}section{background:#fff;padding:24px;margin:22px 0;border-radius:12px}h1,h2{color:#123b50}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;background:#f4f6f8;padding:12px}.note{border-left:4px solid #cc9230;padding-left:14px}a{color:#176d98}</style><h1>跨场景关系归属：适用范围先导</h1><p>4 场景 × 8 模板 × 4 实体 × 2 表达 × 4 字段组合＝1024 次主生成；另加 128 次诊断。Qwen3-8B，目标先输出，原生聊天、禁用思考、贪心生成。</p>']
p.append('<section><h2>本轮发现</h2><p>按运行前锁定的操作性标准，<b>'+str(s['n_full'])+'/4 个新场景达到完整复现筛选标准</b>。'+('达到“至少两个新场景”的正式扩充门槛。' if s['advance_to_formal'] else '未达到“至少两个新场景”的正式扩充门槛。')+'</p>')
p.append(table(['场景','判定','完整复现比较','字段效应通过比较（含完整）','明确支持正确率'],[[cfg['families'][f]['name'],labels[v['label']],', '.join(v['full_contrasts']) or '无',', '.join(v['field_effect_contrasts']) or '无',f"{v['supported_correct_rate']:.1%}"] for f,v in s['families'].items()]))
p.append('<img src="scenario_rates.png" alt="四场景来源值误填率" style="width:100%;height:auto">')
p.append('<p>图中计数只表示来源值误填，分母保留全部计划样本；科研成果另有 2 次结构失败，不是 256 次全部答对。人物与商品的主实验各 256 次均正确留空；科研成果 254 次正确留空、2 次输出数组。</p><p><b>活动场景存在字段组合效应，但不能认定表达交互稳定复现。</b>例如 D−B 在分句表达下为 +21.9 个百分点，在直接连接下为 +43.8；交互为 −21.9 个百分点，95% 模板区间 [−56.25, +6.25]，仅 3/8 模板与交互均值同号。其他比较也没有满足完整交互标准。直接连接在活动材料中反而有更多来源误填，与旧公司实验方向不同。</p><p>四场景的明确缺失诊断共 64/64 正确留空；明确支持另一值的诊断共 64/64 正确抽取，说明本轮零误填场景并非单纯不论证据都输出 null。</p>')
p.append('<p class="note">本轮全部为探索性先导，没有把已观察结果重新包装成验证集。每场景筛查六个预设比较，区间仅描述不确定性，不提供经多重比较校正的显著性结论。即使达到门槛，正式实验仍需新模板、新实体和跨模型验证。</p></section>')
p.append('<section><h2>比较什么，如何判定</h2><p>A 只抽目标；B 目标＋来源；C 目标＋同类型对照；D 目标＋不同类型对照。同组同表达的四种条件正文和字段定义完全相同，只改变所需字段。全部字段组合中目标均最先输出。</p>')
p.append(table(['场景','目标','来源','同类型对照','不同类型对照'],[[v['name']]+v['fields'] for v in cfg['families'].values()]))
p.append('<p>主指标是“缺少目标证据却把来源值填为目标”的比例。每场景报告 B−A、C−A、D−A、C−B、D−B、D−C；交互＝该比较在分句条件下的差值−直接连接条件下的差值。区间均为按 8 个模板成组抽样 2000 次所得 95% 百分位区间，保留每模板所有配对。</p><ul><li>字段效应：至少一种表达下，绝对差值≥12.5 个百分点，且至少 5/8 模板与平均差值同号。</li><li>完整复现还要求同一个比较的交互绝对值≥12.5 个百分点，且至少 5/8 模板同号；允许不同场景有不同方向。</li><li>通过字段效应的表达中，来源误填占有效非空、非字符串 null 目标输出的≥60%；主实验协议失败率≤5%。</li><li>充分证据对照总计至少 15/16 正确，B、D 各至少 7/8 正确。只有字段效应、没有合格交互，归为部分复现。</li></ul><p>这些阈值是预先锁定的研究范围筛选规则，不是自然规律或显著性检验；表中保留所有比较与模板差值，便于检查阈值依赖。全部样本均为零误填时，经验 bootstrap 区间退化为 [0,0]，不能据此声称总体风险或真实效应严格为零。</p></section>')
for f,v in s['families'].items():
 p.append('<section><h2>'+esc(cfg['families'][f]['name'])+'：'+labels[v['label']]+'</h2>')
 p.append(table(['表达','条件','来源误填','对照值误填','其他非空','真实 null','字符串 null','结构失败','辅助字段正确'],[[forms[x['form']],x['condition'],f"{x['source']}/{x['n']}",x['control'],x['other'],x['null'],x['string_null'],x['invalid'],f"{x['auxiliary_correct']}/{x['n_aux']}" if x['n_aux'] else '—'] for x in v['cells'] if x['form'] in ['weak','direct']]))
 p.append(table(['比较','分句差值 [95% CI] pp','直接差值 [95% CI] pp','交互 [95% CI] pp','交互同号模板','完整标准'],[[x['contrast'],pp(x['weak']),pp(x['direct']),pp(x['interaction']),str(x['interaction']['same_sign_templates'])+'/8','通过' if x['full'] else '未通过'] for x in v['contrasts']]))
 p.append('<p>来源值占有效非空错误：'+ '；'.join(forms[k]+f" {x['source']}/{x['nonnull_nonencoding']}（{x['share']:.1%}）" for k,x in v['source_shares'].items())+'。主实验协议失败率 '+f"{v['main_protocol_failure_rate']:.1%}"+'。</p>')
 p.append(table(['诊断','条件','目标正确','来源误填','真实 null','字符串 null','结构失败'],[[forms[x['form']],x['condition'],f"{x['target_correct']}/{x['n']}",x['source'],x['null'],x['string_null'],x['invalid']] for x in v['cells'] if x['form'] not in ['weak','direct']]))
 p.append('<details><summary>所有比较的逐模板差值（不筛选模板）</summary>')
 for x in v['contrasts']:
  p.append('<h3>'+esc(x['contrast'])+'</h3>'+table(['模板','分句 pp','直接 pp','交互 pp'],[[t,f"{100*x['weak']['template_values'][t]:+.1f}",f"{100*x['direct']['template_values'][t]:+.1f}",f"{100*x['interaction']['template_values'][t]:+.1f}"] for t in sorted(x['weak']['template_values'])]))
 p.append('</details><details><summary>每模板第一组材料及 A/B/C/D 输出（包括未复现结果）</summary>')
 for t in range(1,9):
  gid=f'{f}_t{t:02d}_e1';rr=[r for r in rows if r['group_id']==gid and r['evidence']=='main']
  p.append('<h3>'+gid+'</h3>')
  for form in ['weak','direct']:
   subset=sorted([r for r in rr if r['form']==form],key=lambda r:r['condition']);p.append('<p>'+forms[form]+'</p><pre>'+esc(subset[0]['material'])+'</pre>'+table(['条件','原始输出'],[[r['condition'],r['raw_output']] for r in subset]))
 p.append('</details></section>')
p.append('<section><h2>核验与局限</h2><p>'+esc(f"1152 条输出完整且唯一，逐条重新评分一致；全部运行前锁定的数据与评分代码哈希一致。JSON 结构失败 {audit['invalid']}；目标字符串 null {audit['string_null']}；任何字段协议失败 {audit['protocol_failures']}。")+'</p><p>32 个模板为手写独立措辞，模板间更换叙述、信息顺序和句法。直接／分句差别仍是一组句法变化，不能解释为单独操纵“明确程度”。科研成果场景中的机构实体与城市实体不同，任务自身难度也可能不同。每场景定义中都显式提醒几类关系不能互相替代；与旧公司实验并非逐词相同的提示词，不能把跨轮差异全部归因于关系类型。本轮不做内部机制扫描，不以某场景是否复现来删除数据。</p><p>两条格式失败原样保留：research_t04_e2_direct_A 输出数组 ["霁禾研究所"]；research_t04_e4_direct_A 输出数组 [""]。前者提到了材料中的数据提供机构，但由于输出不是要求的目标字段 JSON 对象，未计入主指标，也未悄悄修复为有效答案。</p><p><a href="summary.json">完整汇总与模板区间</a> · <a href="outputs.jsonl">全部逐条输出</a> · <a href="../data/materials.csv">材料表 CSV</a> · <a href="../data/jobs.json">全部提示词与金标准</a> · <a href="../protocol.json">运行前协议</a> · <a href="completion_audit.json">完成核验</a> · <a href="environment.json">模型环境</a></p></section></html>')
(ROOT/'results/report.html').write_text(''.join(p),encoding='utf-8')
print(json.dumps(dict(audit=audit,classification={f:dict(label=v['label'],full=v['full_contrasts'],field=v['field_effect_contrasts']) for f,v in s['families'].items()},n_full=s['n_full']),ensure_ascii=False,indent=2))

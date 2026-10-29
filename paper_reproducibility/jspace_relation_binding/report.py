"""Report held-out template replication, with city and null errors distinguished."""
import csv
import html
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def esc(x):return html.escape(str(x))
def cell(s,f,c,o='average'):return next(x for x in s['cells'] if x['form']==f and x['condition']==c and x['order']==('only' if c=='A' else o))
def contrast(s,f='weak',c='D-B',o='average'):return next(x for x in s['contrasts'] if x['form']==f and x['contrast']==c and x['order']==o)
def fmt(v):return f"{100*v['estimate']:+.1f} [{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}]"
def rate(c,m):
 v=c[m];return f"{v['numerator']}/{v['denominator']} ({100*v['estimate']:.1f}%)"
def table(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def internal_section(cfg):
 if not (OUT/'lens_summary.json').exists():return ''
 summary=read('results/lens_summary.json');selected=summary['selected_layers'];stats=summary['paired_lexical_differences']
 fig,axes=plt.subplots(1,3,figsize=(13,4.4),sharey=True)
 for ax,pos in zip(axes,['P0','PN','P1']):
  for method,color in [('J-lens','#b3514f'),('Logit Lens','#218b88')]:
   rr=sorted([r for r in stats if r['split']=='development' and r['position']==pos and r['method']==method],key=lambda r:r['layer'])
   ax.plot([r['layer'] for r in rr],[r['mean_D_minus_B'] for r in rr],marker='o',label=method,color=color)
  for l in selected:ax.axvline(l,color='#8b98a6',alpha=.4,linestyle=':')
  ax.axhline(0,color='#667788',linewidth=1);ax.set_title(pos);ax.set_xlabel('Zero-indexed block output');ax.grid(axis='y',alpha=.15)
 axes[0].set_ylabel('D-B of office-minus-null lexical logit');axes[2].legend(frameon=False,fontsize=9)
 fig.suptitle('Development-only layer selection; vertical lines mark frozen layers',fontsize=12)
 fig.text(.5,.01,'Descriptive means. P0/PN: 18 groups, both orders. P1: 6 first-entity groups, both orders.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.04,1,.93]);fig.savefig(OUT/'lens_development.png');plt.close(fig)
 tab=[[r['position'],r['layer'],r['method'],f"{r['mean_D_minus_B']:+.3f}",r['n_pairs']] for r in stats if r['split']=='validation']
 raw=[json.loads(x) for x in (OUT/'lens_readouts.jsonl').read_text(encoding='utf-8').splitlines()]
 errors=[]
 for l in selected:
  for method in ['J-lens','Logit Lens']:
   for category in ['json_null','office_city']:
    rr=[r for r in raw if r['split']=='validation' and r['position']=='PN' and r['layer']==l and r['method']==method and r['target_category']==category]
    errors.append([l,method,'自然输出null' if category=='json_null' else '自然误填办公城市',len(rr),f"{np.mean([r['office_minus_null'] for r in rr]):+.3f}" if rr else '无样本'])
 values=[json.loads(x) for x in (OUT/'controlled_value_scores.jsonl').read_text(encoding='utf-8').splitlines()]
 lookup={(r['job_id'],r['candidate']):r for r in values};value_table=[];aggregate=[]
 for form in ['weak','direct','missing','supported']:
  for cond in ['B','D']:
   for order in ['first','last','average']:
    rr=[r for r in values if r['split']=='validation' and r['relation_form']==form and r['condition']==cond and r['candidate']=='office' and (order=='average' or r['target_order']==order)]
    margins=[r['logp']-lookup[r['job_id'],'null']['logp'] for r in rr]
    rec=dict(form=form,condition=cond,order=order,n=len(rr),office_preferred=sum(x>0 for x in margins),mean_logp_margin=float(np.mean(margins)))
    aggregate.append(rec)
    if order=='average':value_table.append([cfg['forms'][form],cond,f"{rec['office_preferred']}/{len(rr)}",f"{rec['mean_logp_margin']:+.3f}"])
 (OUT/'controlled_value_summary.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2),encoding='utf-8')
 return f'''<section><h2>行为复现后：有限内部读取</h2><p>复用同一模型路径的官方 Jacobian lens，文件及模型配置指纹通过核验；lens 来自 100 段独立普通英文文本，校准长度上限 128。当前中文任务的校准质量没有额外保证，因此低读数不能解释为信息不存在。六个城市均为单 token，没有依结果筛选城市。</p><p>P0 是生成 JSON 前的最后提示 token；PN 是重放自然生成的精确 token 前缀，在总部字段冒号后、值内容之前读取；P1 是强制正确前置输出的规范 JSON 前缀。PN 成功定位 {summary['natural_coverage']}/{summary['natural_requested']} 条。P0/PN 覆盖所有分句 B/D 样本；P1 使用每个模板的第一组实体，覆盖分句、直接、明确缺失及充分证据四种材料。</p><p>内部分析方案在行为结果之后、任何内部读数之前锁定。探索模板每隔四层读取，仅按探索 PN 的 D−B 词汇分差绝对值选择两个层：{selected}；随后验证模板只读取这两个层及最终层参考。共 {summary['n_contexts']} 个读取上下文、{summary['n_readouts']} 条层/方法记录，没有头部扫描或激活干预。</p><img src="lens_development.png" alt="探索集J-lens与Logit Lens逐层对比">{table(['位置','层','方法','验证 D−B 词汇分差','配对数'],tab)}<p>上表数值为“办公城市 logit−null logit”的 D−B 差，不是误填概率或关系证据分数。验证 P0 未出现同向正差，而 PN/P1 出现正差；这说明读数随输出上下文而变化，不能据此定位信息首次产生的位置。</p>{table(['层','方法','按自然结果分组','运行数','平均办公城市−null词汇分差'],errors)}<p class="small">正确/错误分组为观察性描述，混合了字段条件与模板差异，没有用这一分组筛选总体样本，也不能当作因果比较。</p></section>
 <section><h2>完整目标值的受控前缀诊断</h2><p>为避免把“城市 token 可读取”误当成“模型决定填城市”，在 P1 上另计算两个完整候选值的最终模型条件对数概率：带引号的办公城市和 JSON null，均带相同的后续逗号或右括号。正分差表示这个特定前缀下更偏向城市。这里比较的是规定的两个候选值，不是自然生成正确率或全部可能答案的归一化概率。</p>{table(['材料','条件','更偏向办公城市','平均城市−null对数概率'],value_table)}<p>分句形式 B 为 1/12、D 为 7/12；直接连接和明确缺失均为 0/12，充分证据均为 12/12。前置辅助字段已使用金标准，说明在这种受控序列中差异仍可出现。强制前缀本身会改变计算过程，因此这些结果与自然生成分开解释。</p><p>当前可支持“关系表达与字段要求共同改变误配行为，并伴随目标值相关表征差异”。尚未证明存在单独的总部关系方向、证据充分性方向，或某个具体部件的因果作用。</p><p><a href="lens_protocol.json">内部读取协议</a> · <a href="lens_layer_selection.json">探索选层记录</a> · <a href="lens_token_audit.json">分词与lens核验</a> · <a href="lens_readouts.jsonl">全部内部读数</a> · <a href="controlled_value_scores.jsonl">完整值条件得分</a> · <a href="controlled_value_summary.json">受控前缀汇总</a></p></section>'''
def main():
 cfg=read('protocol.json');dev=read('results/development_summary.json');val=read('results/validation_summary.json');env=read('results/environment_main.json')
 rows=[json.loads(x) for split in ['development','validation'] for x in (OUT/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
 gate=dict(val['continuation']);gate['development_primary']=dev['continuation']['primary'];gate['development_same_direction']=dev['continuation']['primary']['estimate']>0
 gate['pass']=gate['local_gate'] and gate['development_same_direction'];gate['decision']='proceed_to_limited_lens' if gate['pass'] else 'stop_before_lens'
 (OUT/'gate_decision.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding='utf-8')
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
 forms=['weak','direct','label'];conditions=['A','B','C','D'];colors=['#63758a','#218b88','#c7903f','#b3514f']
 fig,ax=plt.subplots(figsize=(11,5))
 for i,(c,col) in enumerate(zip(conditions,colors)):
  vs=[cell(val,f,c)['office_city_misuse'] for f in forms];y=np.array([v['estimate'] for v in vs])*100
  lo=np.array([v['ci95'][0] for v in vs])*100;hi=np.array([v['ci95'][1] for v in vs])*100
  x=np.arange(3)+(i-1.5)*.19;ax.bar(x,y,.18,label=c,color=col)
  ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='none',color='#243448',capsize=3)
 ax.set_xticks(range(3),['Separate clauses','Direct relation','Explicit label']);ax.set_ylim(0,100);ax.set_ylabel('Office city misassigned to headquarters (%)');ax.set_title('Held-out templates: six templates, three companies each');ax.legend(frameon=False,ncol=4);ax.grid(axis='y',alpha=.15)
 fig.text(.5,.01,'A: HQ only; B: office + HQ; C: launch + HQ; D: year + HQ. 95% template-cluster bootstrap.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'relation_forms.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
 for ax,s,title in zip(axes,[dev,val],['Development templates','Held-out templates']):
  for i,(f,col) in enumerate(zip(forms,['#b3514f','#218b88','#63758a'])):
   vals=contrast(s,f)['template_values'];ts=sorted(vals)
   ax.scatter(np.arange(len(ts))+(i-1)*.12,[100*vals[t] for t in ts],label=f,color=col,s=35)
  ax.set_xticks(range(len(ts)),ts);ax.axhline(0,color='#667788',linewidth=1);ax.set_title(title);ax.set_ylim(-105,105);ax.set_ylabel('D minus B per template (percentage points)');ax.grid(axis='y',alpha=.15)
 axes[1].legend(frameon=False);fig.text(.5,.01,'Descriptive template means: three companies x two output orders per B/D condition.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'template_effects.png');plt.close(fig)
 mainrows=[]
 for f in forms:
  mainrows.append([cfg['forms'][f],*[rate(cell(val,f,c),'office_city_misuse') for c in conditions],fmt(contrast(val,f)),fmt(contrast(val,f,'C-B'))])
 diag=[]
 for f in ['missing','supported']:
  for c in conditions:
   cc=cell(val,f,c);diag.append([cfg['forms'][f],c,rate(cc,'office_city_misuse') if f=='missing' else rate(cc,'supported_correct'),rate(cc,'string_null'),rate(cc,'format_failure'),rate(cc,'protocol_failure')])
 orders=[[cfg['forms'][f],fmt(contrast(val,f,o='first')),fmt(contrast(val,f,o='last')),fmt(contrast(val,f))] for f in forms]
 detailed=[]
 for f in forms:
  for c in conditions:
   cc=cell(val,f,c);detailed.append([cfg['forms'][f],c,rate(cc,'office_city_misuse'),rate(cc,'launch_city_misuse'),rate(cc,'other_string'),rate(cc,'string_null'),rate(cc,'reliable_null'),rate(cc,'format_failure'),rate(cc,'auxiliary_correct') if c!='A' else '不适用'])
 templaterows=[]
 for t in sorted(contrast(val)['template_values']):
  templaterows.append([t,*[f"{100*contrast(val,f)['template_values'][t]:+.1f}" for f in forms]])
 groups=read('data/groups.json');samples=''
 for g in groups:
  if g['split']=='validation' and g['group_id'].endswith('e1'):
   samples+=f"<details><summary>{g['template_id']} · {esc(g['entity'])}</summary>"+''.join(f"<p><b>{cfg['forms'][f]}</b>：{esc(v['text'])}</p>" for f,v in g['variants'].items())+'</details>'
 errors=[r for r in rows if r['split']=='validation' and (r['office_city_misuse'] or r['launch_city_misuse'] or r['other_string'] or r['protocol_failure'] or r['supported_correct'] is False)]
 for r in errors[:12]:samples+=f"<details><summary>错误：{esc(r['job_id'])} · {esc(r['target_category'])}</summary><p>{esc(r['material'])}</p><p>字段：{esc('、'.join(r['fields']))}；目标金标准：{esc(r['gold_value'])}</p><pre>{esc(r['raw_output'])}</pre></details>"
 if not errors:samples+='<p>验证集未观察到上述错误；全部记录仍保留。</p>'
 flat=[]
 for split,s in [('development',dev),('validation',val)]:
  for c in s['cells']:
   for m,v in c.items():
    if isinstance(v,dict) and 'estimate' in v:flat.append(dict(split=split,form=c['form'],condition=c['condition'],order=c['order'],metric=m,**v))
 for name,records in [('behavior_summary.csv',flat),('paired_contrasts.csv',[dict(split=split,**c) for split,s in [('development',dev),('validation',val)] for c in s['contrasts']])]:
  with (OUT/name).open('w',encoding='utf-8-sig',newline='') as h:
   w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 conclusion='分句表达下，年份联合相对办公室联合的关系误配在新模板中达到预定复现标准。' if gate['pass'] else '年份联合诱发总部误填的现象，未达到预定的跨模板复现标准。'
 lens=('已完成有限 J-lens／Logit Lens 与受控目标值诊断；未做因果干预，不能宣称机制已成立。' if (OUT/'lens_summary.json').exists() else '行为门槛通过；需按弱绑定范围开展有限内部读取，不能直接宣称机制已成立。') if gate['pass'] else '按锁定标准停在行为辨别阶段，不启动 J-lens 或头部扫描。'
 internals=internal_section(cfg)
 old=read('data/prior_observation_audit.json')
 body=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>字段组合与关系归属：模板辨别实验</title><style>
 body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f0f4f7;color:#23384b;margin:0;line-height:1.8}}main{{max-width:1200px;margin:auto;padding:28px}}section,.hero{{background:white;border:1px solid #dbe3e9;border-radius:12px;padding:24px;margin:22px 0}}.hero{{border-left:5px solid #218b88}}h1{{font-size:30px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce4ea;vertical-align:top}}th{{background:#edf3f6}}.scroll{{overflow-x:auto}}img{{width:100%;height:auto}}details{{background:#f3f6f8;padding:12px;margin:12px 0}}pre{{white-space:pre-wrap;word-break:break-word}}.small{{color:#5b6d7d;font-size:13px}}a{{color:#087775}}@media(max-width:700px){{main{{padding:12px}}section{{padding:14px}}}}</style></head><body><main>
 <p class="small">新实验 · Qwen3-8B · 12 个模板 × 3 组实体 · 按模板独立划分</p><h1>字段组合如何影响办公室与总部的关系归属</h1>
 <div class="hero"><b>{conclusion}</b><p>验证主要差值（分句表达 D−B）为 {fmt(gate['primary'])} pp；正向模板数 {len(gate['positive_templates'])}/6。探索集为 {fmt(gate['development_primary'])} pp。{lens}</p></div>
 <section><h2>先核实原始观察</h2><p>对原验证输出进行只读核验，原公司模板 A 0/6、办公室联合 B 0/12、年份联合 C 12/12，错误均是把办公室城市填成总部。该旧观察仅作为新实验的动机，不合并进入新验证集。</p><p><b>本轮编号有所变化：</b>A 总部单字段；B 办公城市＋总部；C 发布会城市＋总部；D 成立年份＋总部。旧报告中的年份联合 C 对应本轮 D。</p></section>
 <section><h2>设计与冻结</h2><p>手写 12 个叙述模板，每模板 3 组虚构公司；6 个模板用于探索、6 个从未用于探索的模板用于验证。每组有办公室与城市分句、直接连接、明确标注三种表达，成立年份和发布会城市事实始终保留。每种表达运行 A 单字段一次，以及 B/C/D 两种字段顺序，共 7 次。</p><p>主要试验：36×3×7=756 次。每模板另选第一组实体，运行明确缺失和充分证据两种诊断：12×2×7=168 次。正式总计 924 次（探索 462、验证 462）。另用两个独立模板各一组实体完成 70 次试跑，不计正式结果。</p><p>在每组材料内，A/B/C/D 共用完全相同的正文与四项字段定义，仅改变需要输出的字段及顺序。办公室和发布会城市始终不同，办公城市在两套正式划分中均衡。充分证据对照只覆盖每模板第一实体，城市覆盖窄于主试验。</p></section>
 <section><h2>主要结果：把办公城市填成总部</h2>{table(['关系表达','A 总部','B 办公＋总部','C 发布会＋总部','D 年份＋总部','D−B [95% CI] pp','C−B [95% CI] pp'],mainrows)}<img src="relation_forms.png" alt="按关系表达和抽取条件统计的办公室到总部误配率"><p>主要指标只统计目标精确等于办公室城市，字符串 "null" 不计城市误配。每种主表达下 A 为 18 次、B/C/D 各 36 次；各来自同一批 6 个模板、18 个基础组。</p></section>
 <section><h2>是不是少数模板或字段顺序决定结果</h2>{table(['验证模板','分句 D−B pp','直接 D−B pp','标注 D−B pp'],templaterows)}<img src="template_effects.png" alt="每模板的配对差值">{table(['关系表达','目标前置 D−B','目标后置 D−B','等权平均 D−B'],orders)}<p>区间以模板为单位做 2000 次配对 bootstrap，每次完整保留模板的三组实体、不同表达和字段顺序，不把这些变体当成独立样本。只有六个验证模板，区间和泛化范围均有限；全零时的退化区间不是零风险保证。</p></section>
 <section><h2>错误类型与诊断对照</h2>{table(['表达','条件','办公城市误配','发布会城市误配','其他字符串','字符串null','可靠留空','结构失败','辅助字段正确'],detailed)}<p>“其他字符串”只作待核验类别，不自动宣称是城市幻觉。字符串 "null" 从一开始就单列为编码违规，既不当作实际城市，也不当作可靠留空。结构失败包括解析、键、顺序、类型或截断问题。</p>{table(['诊断材料','条件','误配率／充分证据正确率','字符串null','结构失败','任意规范失败'],diag)}<p>明确缺失和充分证据控制不并入主效应；它们改变了目标证据状态，只用于诊断。</p></section>
 <section><h2>继续标准与解释边界</h2><p>预先规定：验证分句形式的 D−B 至少 +10 pp、模板 bootstrap 区间下界为正、至少 4/6 模板同向，探索总体方向为正，且 D−B 的规范失败增量小于 5 pp。当前增量为 {fmt(gate['protocol_failure_excess'])} pp。{lens}</p><p>如果效应只出现在分句形式，最多支持弱关系绑定这一收窄方向。办公室城市可被读取，不等于总部关系得到证据支持；内部读数与误填相关也不等于因果作用。</p><p>本轮新增发布会事实、相应定义和企业名称标识，与旧试验的提示上下文不同；若不复现，不能单独归因于措辞，也不能抹去旧模板的原始观察。模板虽为独立编写，明确标签形式仍语义相似。材料已经进行语义和结构核查，尚未经第二位人工复核。</p></section>
 {internals}
 <section><h2>材料与逐条输出</h2><p>下面展示每个验证模板的第一组实体及各版本；错误按原始保存顺序列出前 12 条。符合错误筛选条件的验证运行共 {len(errors)} 次，全部 462 条均已保留。</p>{samples}</section>
 <section><h2>文件与复现</h2><p><a href="../data/material_audit.csv">全部材料与金标准</a> · <a href="../protocol.json">锁定协议</a> · <a href="../data/lock.json">材料指纹</a> · <a href="prevalidation_seal.json">验证前冻结记录</a> · <a href="../data/prior_observation_audit.json">旧观察核验</a></p><p><a href="pilot.jsonl">试跑</a> · <a href="development.jsonl">探索完整输出</a> · <a href="validation.jsonl">验证完整输出</a> · <a href="behavior_summary.csv">分条件汇总</a> · <a href="paired_contrasts.csv">模板配对差值</a> · <a href="gate_decision.json">继续决策</a></p><p class="small">服务器项目 ./jspace_relation_binding；模型 {esc(env['model_path'])}；BF16，贪心生成，原生模板关闭显式思考，max_new_tokens=128，batch=12。模型指纹 {env['model_revision']}。</p></section>
 <p class="small">报告生成时间 UTC：{time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())}</p></main></body></html>'''
 (OUT/'report.html').write_text(body,encoding='utf-8');print(json.dumps(gate,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

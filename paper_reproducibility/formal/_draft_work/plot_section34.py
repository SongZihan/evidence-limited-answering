from pathlib import Path
import json,sys,statistics,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
jobs={j['job_id']:j for j in json.loads((base/'jobs.json').read_text(encoding='utf-8'))}
layers=[9,13,17,21,25,29,33];groups=collections.defaultdict(list)
for line in (base/'readouts.jsonl').read_text(encoding='utf-8').splitlines():
 r=json.loads(line);j=jobs[r['job_id']]
 if j['split']!='development' or j['evidence']!='absent' or j['form']!='weak' or r['position']!='PN' or r['layer'] not in layers:continue
 assert j['office_city']==j['city_a'] and j['city_a']!=j['city_b']
 groups[(r['method'],j['condition'],r['layer'])].append((r['job_id'],r['scores'][j['city_a']]-r['scores'][j['city_b']]))
stats=[]
for method in ['J-lens','Logit Lens']:
 for c in ['B','D']:
  for l in layers:
   rr=groups[(method,c,l)];assert len(rr)==16
   stats.append(dict(method=method,condition=c,layer=l,n=16,mean=statistics.mean(v for _,v in rr),job_ids=[j for j,_ in rr]))
for lang in ['en']:
 zh=lang=='zh'
 plt.rcParams.update({'font.family':'Noto Sans CJK SC' if zh else 'DejaVu Sans','svg.fonttype':'none','font.size':8.5,'axes.unicode_minus':False,'text.color':'#34332D','axes.labelcolor':'#48463F','xtick.color':'#787368','ytick.color':'#787368'})
 fig=plt.figure(figsize=(7.3,3.05),facecolor='white')
 fig.text(.02,.98,'总部值生成前的城市读数' if zh else 'City readouts immediately before the headquarters value',fontsize=11 if zh else 10.5,fontweight='bold',va='top')
 for i,method in enumerate(['J-lens','Logit Lens']):
  ax=fig.add_axes([.10+i*.48,.24,.37,.49])
  for c,col in [('B','#4B7098'),('D','#BB6856')]:
   rr=[r for r in stats if r['method']==method and r['condition']==c]
   ax.plot(layers,[r['mean'] for r in rr],color=col,marker='o',ms=3.5,lw=1.6,label=c)
  ax.axhline(0,color='#B5AEA1',lw=.7);ax.set_xlim(8,34);ax.set_ylim(-.4,5)
  ax.set_xticks(layers);ax.set_yticks([0,1,2,3,4,5]);ax.grid(axis='y',color='#ECE8E0',lw=.6)
  for s in ['top','right']:ax.spines[s].set_visible(False)
  for s in ['left','bottom']:ax.spines[s].set_color('#B5AEA1')
  ax.set_title(('a  ' if i==0 else 'b  ')+method,loc='left',fontweight='bold',fontsize=10,pad=9)
  ax.set_xlabel('解码层（零起算）' if zh else 'Decoder layer (zero-based)',labelpad=6)
  if i==0:ax.set_ylabel('办公室城市 − 另一候选城市\nlogit差' if zh else 'Office city − alternative city\nLogit contrast',fontsize=8)
 from matplotlib.lines import Line2D
 handles=[Line2D([0],[0],color=col,marker='o',ms=3.5,lw=1.6,label=label) for col,label in [('#4B7098','B：总部＋办公城市' if zh else 'B: HQ + office city'),('#BB6856','D：总部＋成立年份' if zh else 'D: HQ + founding year')]]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.905),frameon=False,ncol=2,fontsize=8.5)
 fig.text(.02,.075,'Qwen3-8B · 探索集 · 总部缺失／分句表达 · 自然PN · 每条曲线16例' if zh else 'Qwen3-8B · Development set · HQ absent / split clauses · Natural PN · n = 16 per curve',fontsize=7.5,color='#817B70')
 fig.text(.02,.02,'正值只表示相对更易读出办公室城市，不表示应填该城市；图中未与null比较。' if zh else 'Positive values favor the office-city readout; they do not justify answering with it. No null contrast.',fontsize=7.1,color='#817B70')
 file=out/f'figure4_internal_readouts_{lang}'
 fig.savefig(file.with_suffix('.png'),dpi=350,facecolor='white');fig.savefig(file.with_suffix('.svg'),facecolor='white');plt.close(fig)
(out/'figure4_readout_data.json').write_text(json.dumps({'scope':'development absent weak PN only; all 16 samples per condition; no correctness filtering','statistics':stats},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([r for r in stats if r['layer']==33],ensure_ascii=False))

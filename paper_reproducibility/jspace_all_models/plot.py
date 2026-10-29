import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text())
data=read('results/comparison.json');rows=data['source_counts'][1:]
names=['Person','Event','Product','Research','Shipment','Film','Artifact','Book','Procurement'];colors=['#667b8d','#c17c40','#258b85']
fig,ax=plt.subplots(figsize=(13,5.8))
for m,label in enumerate(['Qwen3-8B','Qwen3.5-2B','Qwen3.5-27B']):
 vals=[int(r[m+1].split('/')[0]) for r in rows]
 bars=ax.bar([i+(m-1)*.25 for i in range(9)],[100*v/256 for v in vals],width=.23,color=colors[m],label=label)
 for bar,v in zip(bars,vals):
  if v:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,str(v),ha='center',fontsize=8)
ax.set_xticks(range(9),names,rotation=20,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Source-value misattribution (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False);ax.set_title('Nine scope-pilot scenarios | same tasks across models')
fig.text(.5,.015,'Each model/scenario: 256 main runs. Labels are counts. Other-value errors and string null are excluded.\nThese pooled counts are descriptive; replication screening uses paired template contrasts.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.075,1,1]);fig.savefig(ROOT/'results/nine_scenarios.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for m,key in enumerate(['8b','2b','27b']):
 summary=read(f'results/{key}_company_audited_summary.json')['validation']
 for ax,form in zip(axes,['weak','direct']):
  cc=[next(x for x in summary['cells'] if x['form']==form and x['condition']==c and x['order']==('only' if c=='A' else 'average'))['office_city_misuse'] for c in 'ABCD']
  ax.bar([i+(m-1)*.25 for i in range(4)],[100*x['estimate'] for x in cc],width=.23,color=colors[m],label=['Qwen3-8B','Qwen3.5-2B','Qwen3.5-27B'][m])
  ax.set_xticks(range(4),list('ABCD'));ax.set_title('Split relation' if form=='weak' else 'Direct relation');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Office city misattributed as headquarters (%)');axes[1].legend(frameon=False,fontsize=9)
fig.suptitle('Company validation templates | both target orders retained');fig.text(.5,.015,'A: 18 runs per model/form; B/C/D: 36 runs each. Historical template split retained, not fresh confirmation.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(ROOT/'results/company_rates.png',dpi=160);plt.close(fig)
p=ROOT/'results/report.html';text=p.read_text(encoding='utf-8');text=text.replace('<h2>来源值误填总数</h2>','<h2>来源值误填总数</h2><img src="nine_scenarios.png" style="width:100%;height:auto" alt="九场景来源误填率">').replace('<h2>公司场景：原验证标准与交互</h2>','<h2>公司场景：原验证标准与交互</h2><img src="company_rates.png" style="width:100%;height:auto" alt="公司验证条件误填率">');p.write_text(text,encoding='utf-8')

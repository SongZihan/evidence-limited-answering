import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
data=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
blue='#4B7098';red='#BB6856';gray='#817A6B';cmap=LinearSegmentedColormap.from_list('rates',['#FAFAF7','#D7E1EA','#7594B1'])
for lang in ['en']:
 zh=lang=='zh';plt.rcParams.update({'font.family':'Noto Sans CJK SC' if zh else 'DejaVu Sans','svg.fonttype':'none','font.size':8,'axes.unicode_minus':False,'text.color':'#34332D','axes.labelcolor':'#5E594F','xtick.color':'#777165','ytick.color':'#44413A'})
 fig=plt.figure(figsize=(9.4,5.65),facecolor='white')
 ax=fig.add_axes([.225,.20,.29,.63]);bx=fig.add_axes([.555,.20,.18,.63]);cx=fig.add_axes([.78,.20,.20,.63])
 values=[]
 for m in data:
  s=m['strict'];cells=s['cells'];row=[100*cells['weak_'+c]['source']/cells['weak_'+c]['n'] for c in 'ABCD'];row.append(100*sum(cells['weak_'+c]['failure'] for c in 'ABCD')/126);values.append(row)
 ax.imshow(values,cmap=cmap,vmin=0,vmax=100,aspect='auto')
 for i,m in enumerate(data):
  for j,v in enumerate(values[i]):
   invalid=j<4 and m['strict']['cells']['weak_'+'ABCD'[j]]['failure']==m['strict']['cells']['weak_'+'ABCD'[j]]['n']
   ax.text(j,i,'—' if invalid else f'{v:.1f}',ha='center',va='center',fontsize=7.8,color='#34332D')
 ax.set_yticks(range(13),[m['model'] for m in data]);ax.tick_params(axis='both',length=0,pad=6)
 ax.set_xticks(range(5),['A','B','C','D','失败' if zh else 'Fail']);ax.xaxis.tick_top()
 ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,13,1),minor=True);ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',length=0)
 for sp in ax.spines.values():sp.set_visible(False)
 for a,title in [(ax,'a  分句来源误填率（%）' if zh else 'a  Source misuse in split form (%)'),(bx,'b  字段效应 D−B' if zh else 'b  Field effect D−B'),(cx,'c  表达交互' if zh else 'c  Wording interaction')]:
  a.set_title(title,loc='left',fontsize=9.2,fontweight='bold',pad=30)
 for i,m in enumerate(data):
  s=m['strict'];allinvalid=values[i][-1]==100
  if allinvalid:
   for a in [bx,cx]:a.text(0,i,'—',va='center',ha='center',color=gray)
   continue
  bx.plot([s['weak_gap'],s['direct_gap']],[i-.10,i+.10],color='#C7C0B4',lw=.8)
  bx.scatter(s['weak_gap'],i-.10,s=20,color=red,zorder=3);bx.scatter(s['direct_gap'],i+.10,s=22,facecolor='white',edgecolor=blue,zorder=3)
  low,high=s['interaction_ci'];v=s['interaction'];cx.plot([low,high],[i,i],color=gray,lw=1.1);cx.scatter(v,i,s=19,color=blue,zorder=3)
 for a in [bx,cx]:
  a.set_ylim(12.5,-.5);a.set_yticks(range(13),['']*13);a.tick_params(axis='y',length=0);a.set_xlim(-40,75);a.set_xticks([-25,0,25,50]);a.axvline(0,color='#ABA496',lw=.8);a.grid(axis='y',color='#F0EDE7',lw=.5);a.set_xlabel('百分点' if zh else 'Percentage points')
  for k in ['top','right','left']:a.spines[k].set_visible(False)
  a.spines['bottom'].set_color('#B9B2A5')
 fig.legend([Line2D([],[],marker='o',color=red,ls='',ms=4),Line2D([],[],marker='o',color=blue,markerfacecolor='white',ls='',ms=4)],['分句' if zh else 'Split','直接连接' if zh else 'Direct'],loc='upper center',bbox_to_anchor=(.65,.875),frameon=False,ncol=2,fontsize=8)
 fig.text(.02,.985,'相同公司材料下的跨模型比较' if zh else 'Cross-model comparison on identical company passages',ha='left',va='top',fontsize=12,fontweight='bold')
 lines=(['A：仅总部；B：总部＋办公城市；C：总部＋发布会城市；D：总部＋成立年份。',
 '严格评分；公司验证集6模板。A每种表达18条，B/C/D各36条，联合条件平衡字段顺序。',
 '“失败”为分句全部126条的协议失败率；—表示该条件全部协议失败，不作为零误填解释。',
 '交互＝(D−B)分句−(D−B)直接；线段为模板配对bootstrap 95%区间。零经验区间不代表总体风险为零。'] if zh else
 ['A: HQ only; B: HQ + office city; C: HQ + launch city; D: HQ + founding year.',
 'Strict scoring; 6 held-out company templates. Per wording: n=18 for A, n=36 for B/C/D; joint-field order balanced.',
 'Fail: protocol failures among all 126 split-form runs. —: all runs fail the protocol; not evidence of zero misuse.',
 'Interaction = (D−B)split − (D−B)direct. Lines: template-paired bootstrap 95% intervals; zero intervals do not imply zero risk.'])
 for i,line in enumerate(lines):fig.text(.02,.125-i*.031,line,fontsize=7.4 if zh else 6.9,color='#777165',va='top')
 stem='figure5_model_comparison'+('_en' if not zh else '')
 fig.savefig(out/(stem+'.png'),dpi=350);fig.savefig(out/(stem+'.svg'));plt.close(fig)


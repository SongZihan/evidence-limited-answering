import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
old=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
new=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));out=Path(sys.argv[3]);out.mkdir(exist_ok=True,parents=True)
data=[]
for m in old:
 if 'glm' in m['model'].lower():continue
 key={'Qwen3.5-2B':'qwen2b','Qwen3-8B':'qwen8b'}.get(m['model']);n=96 if key else 16
 evs=[new[key]['causal'][e] for e in ['absent','supported','swapped']] if key else [m['validation']['evidence'][e] for e in ['main','supported','swapped']]
 counts=[(e['replay'],e['forward']) if key else (e['before_correct'],e['after_correct']) for e in evs]
 data.append(dict(name=m['model'],n=n,counts=counts,layer=m['selection']['layer'],rank=m['selection']['rank'],key=key))
blue='#4B7098';red='#BB6856';gray='#817A6B';cmap=LinearSegmentedColormap.from_list('gain',[red,'#FAFAF7',blue])
for zh in [False]:
 plt.rcParams.update({'font.family':'Noto Sans CJK SC' if zh else 'DejaVu Sans','svg.fonttype':'none','font.size':8.5,'axes.unicode_minus':False,'text.color':'#393830'})
 fig=plt.figure(figsize=(10,4.6),facecolor='white');a=fig.add_axes([.235,.27,.36,.47]);b=fig.add_axes([.70,.27,.28,.47])
 vals=np.array([[(v-u)/m['n']*100 for u,v in m['counts']] for m in data]);a.imshow(vals,cmap=cmap,vmin=-75,vmax=75,aspect='auto')
 for i,m in enumerate(data):
  for j,(u,v) in enumerate(m['counts']):a.text(j,i,f'{u} → {v}'+('†' if i==0 and j==1 else ''),ha='center',va='center',fontsize=8.5)
 a.set_xticks(range(3),['总部缺失','明确总部','关系交换'] if zh else ['HQ absent','HQ stated','Roles swapped']);a.xaxis.tick_top()
 a.set_yticks(range(len(data)),[f"{m['name']}  L{m['layer']}/r{m['rank']}\nn={m['n']}" for m in data]);a.tick_params(length=0,pad=6,labelsize=8)
 a.set_xticks(np.arange(-.5,3,1),minor=True);a.set_yticks(np.arange(-.5,len(data),1),minor=True);a.grid(which='minor',color='white',lw=1.3);a.tick_params(which='minor',length=0)
 for sp in a.spines.values():sp.set_visible(False)
 a.set_title('a  正确案例数：未干预 → 低维替换' if zh else 'a  Correct cases: untreated → low rank',loc='left',fontsize=9.3,fontweight='bold',pad=29)
 for i,m in enumerate(data[:2]):
  c=new[m['key']]['causal']['absent'];counts=[c[k] for k in ['replay','random','forward','full']];x=np.arange(4)+(i-.5)*.28
  b.bar(x,np.array(counts)/96*100,.25,color=blue if i==0 else red,label=m['name'])
  for xi,n in zip(x,counts):b.text(xi,n/96*100+2,str(n)+'/96',ha='center',fontsize=7)
 b.set_ylim(0,110);b.set_yticks([0,25,50,75,100]);b.set_xticks(range(4),['未干预','随机','低维','完整'] if zh else ['Untreated','Random','Low rank','Full']);b.tick_params(axis='x',labelsize=7)
 b.set_ylabel('正确留空率（%）' if zh else 'Correct null answers (%)');b.set_title('b  扩充验证集的缺失条件对照' if zh else 'b  Expanded validation: HQ absent',loc='left',fontsize=9.3,fontweight='bold',pad=29)
 for sp in ['top','right']:b.spines[sp].set_visible(False)
 for sp in ['left','bottom']:b.spines[sp].set_color('#B5AEA1')
 b.set_axisbelow(True);b.grid(axis='y',color='#ECE8E0',lw=.6);b.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2)
 fig.text(.02,.96,'低维替换的收益取决于能否保留有证据的回答' if zh else 'Low-rank gains must preserve evidence-supported answers',fontsize=11.5,fontweight='bold')
 fig.text(.02,.88,'分句表达 · 相同前缀未干预基线 · Qwen两模型为扩充验证，其余为原验证' if zh else 'Split wording · Matched-prefix baseline · Expanded Qwen samples; original validation for other models',fontsize=8,color=gray)
 foot=('a：Qwen两模型每条件96例，其余每条件16例；颜色按正确率变化着色，数字为正确数。\n† 2B明确总部有1对未定位：95对可干预均正确，另1例保留原正确输出、不计修复。L为零起算层，r为秩。\nb：每组96例；随机对照为一个种子。采用内容评分。原冻结子空间不重新拟合，两模型均未达到18/24模板改善门槛。' if zh else 'a: n=96 per condition for Qwen; n=16 for other models. Color encodes percentage-point change; labels show counts.\n† 2B HQ-stated: 95 eligible pairs remain correct; one unlocated pair retains its correct baseline, with no repair credited. L is zero-based; r is rank.\nb: n=96; one random seed; content scoring. Original bases remain frozen. Neither model meets the 18/24 improved-template threshold.')
 fig.text(.02,.03,foot,fontsize=7 if zh else 6.5,linespacing=1.65,color=gray)
 for ext in ['png','svg']:fig.savefig(out/('figure8_intervention_selectivity'+('' if zh else '_en')+'.'+ext),dpi=350)
 plt.close(fig)

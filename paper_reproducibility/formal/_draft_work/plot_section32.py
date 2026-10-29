"""Render matched Chinese and English scientific figures from row-level records."""
from pathlib import Path
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

src=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(s) for s in src.read_text(encoding='utf-8').splitlines()]
rows=[r for r in rows if r['evidence_version']=='absent' and r['relation_form'] in ['weak','direct']]
templates=sorted({r['template_id'] for r in rows})
stats={}
for form in ['weak','direct']:
 stats[form]={}
 for cond in 'ABCD':
  rr=[r for r in rows if r['relation_form']==form and r['condition']==cond]
  stats[form][cond]={'n':len(rr),'errors':sum(r['office_city_misuse'] for r in rr),'rate':100*np.mean([r['office_city_misuse'] for r in rr])}
  assert all(r['format_valid'] and r['protocol_valid'] for r in rr)
diffs={}
for form in ['weak','direct']:
 diffs[form]=[]
 for t in templates:
  rates=[]
  for cond in 'BD':
   rr=[r for r in rows if r['template_id']==t and r['relation_form']==form and r['condition']==cond]
   assert len(rr)==6
   rates.append(np.mean([r['office_city_misuse'] for r in rr]))
  diffs[form].append(float(100*(rates[1]-rates[0])))
assert len(templates)==6 and all(v>0 for v in diffs['weak'])
for lang in ['en']:
 plt.rcParams.update({'font.family':'Noto Sans CJK SC' if lang=='zh' else 'DejaVu Sans','font.size':9,'axes.edgecolor':'#B5AEA1','axes.labelcolor':'#48463F','text.color':'#34332D','xtick.color':'#787368','ytick.color':'#48463F','svg.fonttype':'none','axes.unicode_minus':False})
 fig=plt.figure(figsize=(7.3,4.0),facecolor='white')
 ax=fig.add_axes([.205,.22,.35,.52]);bx=fig.add_axes([.71,.22,.27,.52])
 orange='#BB6856';blue='#4B7098';neutral='#D8D2C6'
 title='额外字段改变误填率，直接连接表达削弱这一影响' if lang=='zh' else 'Additional fields change errors; direct relation wording attenuates the effect'
 fig.text(.015,.98,title,fontsize=11,fontweight='bold',va='top')
 fig.text(.015,.795,'a',fontsize=11,fontweight='bold',color='#817357')
 fig.text(.045,.795,'四种抽取要求' if lang=='zh' else 'Four extraction requests',fontsize=9.5,fontweight='bold')
 fig.text(.625,.795,'b',fontsize=11,fontweight='bold',color='#817357')
 fig.text(.655,.795,'逐模板的字段差异' if lang=='zh' else 'Field effect by template',fontsize=9.5,fontweight='bold')
 labels=['只抽总部','总部＋办公城市','总部＋发布会城市','总部＋成立年份'] if lang=='zh' else ['Headquarters only','Headquarters +\noffice city','Headquarters +\nlaunch city','Headquarters +\nfounding year']
 for fi,(form,color) in enumerate([('weak',orange),('direct',blue)]):
  yy=np.arange(4)+(-.17 if fi==0 else .17)
  vals=[stats[form][c]['rate'] for c in 'ABCD']
  ax.barh(yy,vals,height=.28,color=color,zorder=3)
  for y,v,c in zip(yy,vals,'ABCD'):
   st=stats[form][c]
   if not v:ax.plot(0,y,'o',color=color,ms=3,zorder=4,clip_on=False)
   ax.text(v+1.6,y,f"{v:.1f}% ({st['errors']}/{st['n']})",va='center',fontsize=8.5,color=color)
 ax.set_yticks(range(4),labels,fontsize=8.5);ax.set_ylim(3.6,-.6);ax.set_xlim(0,100)
 ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('办公室城市误填率（%）' if lang=='zh' else 'Office-city misassignment (%)',labelpad=9)
 ax.tick_params(axis='y',length=0,pad=8);ax.grid(axis='x',color='#ECE8E0',lw=.6,zorder=0)
 for s in ['top','right','left']:ax.spines[s].set_visible(False)
 # Every row is the same template under the two relation forms.
 for i,(wv,dv) in enumerate(zip(diffs['weak'],diffs['direct'])):
  bx.plot([dv,wv],[i,i],color=neutral,lw=2,zorder=1)
  bx.scatter(wv,i,color=orange,s=40,zorder=3)
  bx.scatter(dv,i,color=blue,s=38,marker='D',edgecolor='white',linewidth=.5,zorder=4)
 bx.axvline(0,color='#B5AEA1',lw=.8,zorder=0)
 bx.set_yticks(range(6),[f'模板 {i+1}' for i in range(6)] if lang=='zh' else [f'Template {i+1}' for i in range(6)])
 bx.set_ylim(5.5,-.5);bx.set_xlim(-4,103);bx.set_xticks([0,25,50,75,100]);bx.tick_params(axis='y',length=0,pad=6)
 bx.set_xlabel('成立年份联合 − 办公城市联合\n误填率差（百分点）' if lang=='zh' else 'Founding year − office city\nMisassignment difference (pp)',fontsize=8,labelpad=7)
 for s in ['top','right','left']:bx.spines[s].set_visible(False)
 bx.grid(axis='x',color='#ECE8E0',lw=.6,zorder=0)
 from matplotlib.lines import Line2D
 fig.legend(handles=[Line2D([0],[0],marker='o',color=orange,lw=2,label='分句表达' if lang=='zh' else 'Split-clause wording'),Line2D([0],[0],marker='D',color=blue,lw=2,label='直接连接' if lang=='zh' else 'Direct relation wording')],loc='upper center',bbox_to_anchor=(.50,.915),frameon=False,fontsize=8.5,ncol=2)
 fig.text(.015,.025,'Qwen3-8B；总部证据缺失；六个验证模板；联合抽取合并目标字段前置、后置结果。' if lang=='zh' else 'Qwen3-8B · Headquarters evidence absent · Six validation templates · Both target-field orders pooled',fontsize=7.2,color='#827D72')
 # The legend sits above the axes; avoid title/legend overlap with a shorter title line.
 if lang=='en':fig.texts[0].set_fontsize(9.8)
 base=out/f'figure3_behavior_effect_{lang}'
 fig.savefig(base.with_suffix('.png'),dpi=350,facecolor='white')
 fig.savefig(base.with_suffix('.svg'),facecolor='white')
 plt.close(fig)
audit={'source':str(src),'stats':stats,'template_ids_in_plot_order':templates,'template_differences_pp':diffs,'split_gap_pp':float(np.mean(diffs['weak'])),'direct_gap_pp':float(np.mean(diffs['direct'])),'interaction_pp':float(np.mean(diffs['weak'])-np.mean(diffs['direct']))}
(out/'figure3_data.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False))

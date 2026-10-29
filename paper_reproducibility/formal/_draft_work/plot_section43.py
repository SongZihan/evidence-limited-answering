import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
data=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
blue='#4B7098';red='#BB6856';gray='#817A6B'
for lang in ['en']:
 zh=lang=='zh';plt.rcParams.update({'font.family':'Noto Sans CJK SC' if zh else 'DejaVu Sans','svg.fonttype':'none','font.size':8,'axes.unicode_minus':False,'text.color':'#393830','axes.labelcolor':'#393830'})
 fig=plt.figure(figsize=(10,4.1),facecolor='white')
 axes=[fig.add_axes([.07,.29,.25,.46]),fig.add_axes([.40,.29,.23,.46]),fig.add_axes([.72,.29,.26,.46])]
 for ax in axes:
  for s in ['top','right']:ax.spines[s].set_visible(False)
  for s in ['left','bottom']:ax.spines[s].set_color('#B5AEA1')
  ax.grid(axis='y',color='#ECE8E0',lw=.6);ax.tick_params(length=3,color='#817A6B')
 a,b,c=axes
 for pos,col in [('P0',blue),('PN',red)]:
  for method,ls in [('J-lens','-'),('Logit Lens','--')]:
   rows=[r for r in data['curves'] if r['position']==pos and r['method']==method]
   a.plot([r['layer'] for r in rows],[r['mean'] for r in rows],color=col,ls=ls,lw=1.5,marker='o' if method=='J-lens' else None,ms=3,label=pos+' · '+method)
 a.axhline(0,color='#B5AEA1',lw=.7);a.set_xticks([9,17,25,33]);a.set_xlabel('解码层（零起算）' if zh else 'Decoder layer (zero-based)');a.set_ylabel('城市读数差的D−B变化' if zh else 'D−B shift in city contrast');a.set_title('a  指令差异随读取位置变化' if zh else 'a  Readout position matters',loc='left',fontsize=9.4,fontweight='bold',pad=12)
 a.legend(fontsize=6.8,frameon=False,loc='upper left',ncol=1)
 for method,col,ls in [('J-lens',blue,'-'),('Logit Lens',red,'--')]:
  r=next(r for r in data['swaps'] if r['condition']=='D' and r['method']==method)
  b.plot([0,1],[r['before_mean'],r['after_mean']],color=col,ls=ls,lw=1.8,marker='o',ms=4,label=method)
 b.axhline(0,color='#B5AEA1',lw=.7);b.set_xlim(-.25,1.25);b.set_ylim(-8,8);b.set_yticks([-6,0,6]);b.set_xticks([0,1],['总部＝乙','总部＝甲'] if zh else ['HQ = B','HQ = A']);b.set_ylabel('城市甲 − 城市乙（logit）' if zh else 'City A − city B (logit)');b.set_title('b  交换关系后读数反转' if zh else 'b  Swapping the relations',loc='left',fontsize=9.4,fontweight='bold',pad=12);b.legend(frameon=False,fontsize=7,loc='upper left')
 for i,form in enumerate(['weak','direct']):
  for cond,col,offset in [('B',blue,-.15),('D',red,.15)]:
   r=next(r for r in data['candidates'] if (r['evidence'],r['form'],r['condition'])==('absent',form,cond));xs=[i+offset+(k%4-1.5)*.025 for k in range(16)]
   c.scatter(xs,[v*100 for v in r['values']],s=13,color=col,alpha=.4,edgecolor='none');c.plot([i+offset-.08,i+offset+.08],[r['mean']*100]*2,lw=2.3,color=col)
   c.text(i+offset,r['mean']*100+4,f"{r['mean']*100:.1f}",ha='center',fontsize=7.5,color=col)
 c.set_ylim(-4,115);c.set_xlim(-.5,1.5);c.set_xticks([0,1],['分句','直接连接'] if zh else ['Split','Direct']);c.set_yticks([0,25,50,75,100]);c.set_ylabel('候选集合内 null 概率（%）' if zh else 'Candidate-set null probability (%)');c.set_title('c  完整候选答案比较' if zh else 'c  Complete answer candidates',loc='left',fontsize=9.4,fontweight='bold',pad=12)
 c.legend(handles=[Line2D([0],[0],color=blue,lw=2,label='B'),Line2D([0],[0],color=red,lw=2,label='D')],frameon=False,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,fontsize=7)
 fig.text(.02,.96,'内部读数随指令、关系归属和生成上下文变化' if zh else 'Internal readouts depend on instructions, relations, and output context',fontsize=12 if zh else 11,fontweight='bold')
 fig.text(.02,.88,'Qwen3-8B · 独立验证集 · 每组16例 · 不按输出正确性筛选' if zh else 'Qwen3-8B · Held-out validation · n = 16 per group · No outcome-based filtering',fontsize=8,color=gray)
 foot=('a：总部缺失／分句；先算“办公室城市−另一城市”，再作D−B。P0为JSON前，PN为自然总部值前。\nb：分句D、自然PN、第33层；固定城市甲乙身份，仅交换办公室与总部归属；曲线为配对样本均值。\nc：总部缺失、统一P1前缀；点为单例，横线为均值。仅在null和两个带引号城市的完整续写间归一化。' if zh else 'a: HQ absent / split. Contrast = office city − alternative, then D−B. P0: before JSON; PN: before the natural HQ value.\nb: Split / D / natural PN / layer 33. City identities stay fixed; office and HQ roles swap. Lines show paired-group means.\nc: HQ absent / fixed P1 prefix. Dots: individual cases; bars: means. Normalize only over null and two quoted city continuations.')
 fig.text(.02,.045,foot,fontsize=7.3 if zh else 6.9,linespacing=1.7,color=gray)
 name='figure7_validation_readouts'+('_en' if not zh else '')
 for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=350)
 plt.close(fig)

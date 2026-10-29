import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
data=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
families=['company','person','event','product','research','shipment','film','artifact','book','procurement']
zhlabels=['公司\n总部城市','人物\n出生城市','活动\n举办城市','商品\n生产城市','科研\n资助机构','物流\n收货城市','电影\n故事城市','文物\n出土城市','图书\n版权机构','采购\n合同日期']
enlabels=['Company\nHQ city','Person\nBirth city','Event\nVenue city','Product\nOrigin city','Research\nFunder','Shipment\nDestination','Film\nStory city','Artifact\nFindspot','Book\nCopyright','Procurement\nContract date']
cmap=LinearSegmentedColormap.from_list('effect',['#7091B0','#FAFAF7','#C27A66']);cmap.set_bad('#E7E5DF')
for lang in ['en']:
 zh=lang=='zh';plt.rcParams.update({'font.family':'Noto Sans CJK SC' if zh else 'DejaVu Sans','svg.fonttype':'none','font.size':9,'axes.unicode_minus':False,'text.color':'#393830'})
 fig=plt.figure(figsize=(10,6.1),facecolor='white');ax=fig.add_axes([.22,.22,.76,.62])
 vals=np.array([[m['scenarios'][f]['weak_gap'] for f in families] for m in data]); mask=np.zeros_like(vals,dtype=bool);flags=np.zeros_like(mask)
 for i,m in enumerate(data):
  for j,f in enumerate(families):
   c=m['scenarios'][f]['cells'];rates=[c['weak_'+k]['failure']/c['weak_'+k]['n'] for k in 'BD'];mask[i,j]=any(x==1 for x in rates);flags[i,j]=any(x>=.2 for x in rates)
 im=ax.imshow(np.ma.array(vals,mask=mask),cmap=cmap,vmin=-50,vmax=50,aspect='auto')
 for i in range(len(data)):
  for j in range(10):
   label='—' if mask[i,j] else ('0' if vals[i,j]==0 else f'{vals[i,j]:.1f}')+('†' if flags[i,j] else '')
   ax.text(j,i,label,ha='center',va='center',fontsize=8.6,color='#34332D')
 ax.set_yticks(range(13),[m['model'] for m in data]);ax.set_xticks(range(10),zhlabels if zh else enlabels);ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=7)
 if not zh: ax.tick_params(axis='x',labelsize=7.5)
 ax.set_xticks(np.arange(-.5,10,1),minor=True);ax.set_yticks(np.arange(-.5,13,1),minor=True);ax.grid(which='minor',color='white',linewidth=1.2);ax.tick_params(which='minor',length=0)
 for s in ax.spines.values():s.set_visible(False)
 fig.text(.02,.965,'字段效应在不同关系中的分布' if zh else 'Field effects across relations',fontsize=13,fontweight='bold')
 fig.text(.02,.925,'分句表达下的来源误填率差 D−B；全部计划样本为分母' if zh else 'Source-misuse difference D−B in split wording; all scheduled cases retained',fontsize=9,color='#817A6B')
 cax=fig.add_axes([.57,.13,.35,.02]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-50,-25,0,25,50]);cb.outline.set_visible(False);cb.ax.tick_params(length=2,labelsize=8)
 fig.text(.22,.145,'蓝：D低于B　　红：D高于B（百分点）' if zh else 'Blue: D < B     Red: D > B (pp)',fontsize=8.6)
 foot=('B：目标＋来源字段；D：目标＋不同类型对照字段。公司每条件36例，其余每条件32例。\n灰色横线：B或D全部协议失败。†：至少一组失败率≥20%，仅提示输出质量，不作显著性标记。\n零差值不等于没有误填，也不排除A、C条件下的指令效应。' if zh else 'B: target + source field; D: target + different-type control. n=36 per company condition; n=32 elsewhere.\nGray dash: B or D entirely fails protocol. †: failure rate ≥20% in either group; a quality flag, not significance.\nZero difference does not imply zero errors or exclude instruction effects involving A or C.')
 fig.text(.02,.025,foot,fontsize=8.1,linespacing=1.65,color='#817A6B')
 base='figure6_scenario_effects'+('_en' if not zh else '')
 for ext in ['png','svg']:fig.savefig(out/(base+'.'+ext),dpi=350)
 plt.close(fig)

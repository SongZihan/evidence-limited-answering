"""Apply the publication layout to freshly plotted English SVGs."""
from pathlib import Path
from lxml import etree as E
import re,sys
OUT=Path(sys.argv[1]);S='http://www.w3.org/2000/svg';q=lambda x:'{'+S+'}'+x
NAMES=['instruction_example','behavior_effect','internal_readouts','model_comparison','scenario_effects','validation_readouts','intervention_selectivity']
report=[]
for n,name in enumerate(NAMES,1):
 if n==1:continue  # Diagram is maintained directly as an editable SVG.
 p=OUT/f'figure{n}_{name}_en.svg'
 tree=E.parse(str(p));r=tree.getroot();removed=[];moved=[]
 def drop(e):
  removed.append(''.join(e.itertext()));e.getparent().remove(e)
 def move(e,x,y):
  moved.append(''.join(e.itertext()));e.attrib.pop('transform',None);e.set('x',str(x));e.set('y',str(y))
  style=e.get('style','');style=re.sub(r'text-anchor:\s*\w+','text-anchor: middle',style);e.set('style',style+';text-anchor:middle')
 texts=list(r.iter(q('text')))
 if n==1:
  for e in texts:
   t=''.join(e.itertext());y=float(e.get('y','0'))
   if y<110 or y>1020:drop(e)
   elif t in ['a','b','c']:move(e,24+516*['a','b','c'].index(t),988);e.set('style','text-anchor:start')
   elif t in ['Original condition','Change requested fields','Rephrase the relation']:
    i=['Original condition','Change requested fields','Rephrase the relation'].index(t);move(e,64+516*i,986);e.set('style','text-anchor:start')
   elif y>=541:e.set('y',str(y-60))
  for e in r.findall(q('rect')):
   if e.get('y') and float(e.get('y'))>541:e.set('y',str(float(e.get('y'))-60))
  for e in list(r.findall(q('path'))):
   d=e.get('d','')
   if '1012' in d or '113' in d:r.remove(e)
   elif '492' in d:e.set('d',d.replace('492','485').replace('990','950'))
  box=(0,125,1560,880)
 else:
  cfg={2:(20,282,302),3:(22,214,228),4:(48,374,390),5:(34,None,398),6:(56,272,288),7:(65,302,317)}[n]
  top,py,bottom=cfg
  for e in texts:
   t=''.join(e.itertext());y=float(e.get('y','-999'))
   # Remove only named explanatory blocks; legends and numerical data remain.
   kill=(n==2 and (t.startswith('Additional fields change') or t.startswith('Qwen3-8B ·')) or
    n==3 and (t.startswith('City readouts immediately') or t.startswith('Qwen3-8B ·') or t.startswith('Positive values favor')) or
    n==4 and (t.startswith('Cross-model comparison') or t.startswith(('A: HQ only;','Strict scoring;','Fail: protocol','Interaction ='))) or
    n==5 and t.startswith(('Field effects across','Source-misuse difference','B: target +','Gray dash:','Zero difference')) or
    n==6 and t.startswith(('Internal readouts depend','Qwen3-8B ·','a: HQ absent','b: Split /','c: HQ absent')) or
    n==7 and t.startswith(('Low-rank gains','Split wording ·','a: n=','† 2B HQ-stated:','b: n=')))
   if kill:drop(e);continue
   if n==2:
    labels={'a':(160,py),'Four extraction requests':(160,py),'b':(440,py),'Field effect by template':(440,py)}
    if t in ['a','b']:drop(e)
    elif t in labels:
     e.text=('a  ' if t.startswith('Four') else 'b  ')+t;move(e,*labels[t])
   elif re.match('^[abc]  ',t):
    centers={3:[150,402],4:[245,439,589],6:[140,371,612],7:[250,605]}[n];move(e,centers[ord(t[0])-97],py)
    if n==4 and t.startswith('a '):e.text='a  Source misuse (%)'
  width=float(r.get('viewBox').split()[2]);box=(0,top,width,bottom-top)
 # Preserve paths/data coordinates; tighten canvas by changing its viewBox only.
 r.set('viewBox',' '.join(map(str,box)));r.set('width',str(box[2])+'pt');r.set('height',str(box[3])+'pt')
 bg=E.Element(q('rect'),x=str(box[0]),y=str(box[1]),width=str(box[2]),height=str(box[3]),fill='white');r.insert(0,bg)
 tree.write(str(p),encoding='UTF-8',xml_declaration=True)
 report.append({'figure':n,'removed':removed,'moved':moved,'viewBox':box})

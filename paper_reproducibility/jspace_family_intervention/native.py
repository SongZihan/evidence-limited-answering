import re
def answer_span(key,raw):
 if key=='kimi':return 0,len(raw)-len('<|im_end|>') if raw.endswith('<|im_end|>') else len(raw)
 if key=='muse':
  matches=list(re.finditer(r'(?:^|<\|eom\|><\|start\|>assistant)\s*to=user<\|message\|>',raw))
  if len(matches)!=1:return None
  start=matches[0].end();end=len(raw)
  for marker in ['<|eom|>','<|eot|>']:
   k=raw.find(marker,start)
   if k>=0:end=min(end,k)
  return start,end
 if key=='glm':
  if '<answer>' in raw:
   if raw.count('<answer>')!=1:return None
   start=raw.index('<answer>')+len('<answer>');end=len(raw)
   if raw.endswith('</answer>'):end-=len('</answer>')
   return start,end
  if '</think>' in raw:return raw.index('</think>')+len('</think>'),len(raw)
  if '<think>' in raw:return None
 return 0,len(raw)

def locate(key,raw,field,prefixes):
 span=answer_span(key,raw)
 if span is None:return None,'missing_final_answer'
 start,end=span;body=raw[start:end]
 match=re.search('"'+re.escape(field)+r'"\s*:(\s*)',body)
 if not match:return None,'missing_target_key'
 colon=start+match.start(1);value=start+match.end()
 if value>=end:return None,'missing_target_value'
 # prefixes[k] is the decoded generated prefix of k tokens, including protocol tokens.
 valid=[(n,s) for n,s in enumerate(prefixes) if raw.startswith(s) and len(s)<=value]
 exact=[(n,s) for n,s in valid if len(s)>=colon]
 if exact:n,s=exact[-1];mode='exact'
 elif valid:n,s=valid[-1];mode='preceding_token'
 else:return None,'unstable_token_decoding'
 return dict(n=n,prefix=s,mode=mode,value_char=value,colon_char=colon,regenerated_syntax=raw[len(s):value],answer_start=start,answer_prefix=raw[start:len(s)] if len(s)>=start else ''),None

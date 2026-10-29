# JSON null 与字符串 null 分开计分。 / Distinguish JSON null from its string form.
import json
def score(j,raw,truncated=False):
 issues=[];dups=[]
 def pairs(ps):
  obj={}
  for k,v in ps:
   if k in obj:dups.append(k)
   obj[k]=v
  return obj
 def bad(x):raise ValueError(x)
 try:obj=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
 except ValueError:obj=None;issues.append('invalid_json')
 if not isinstance(obj,dict):issues.append('not_object');obj={}
 if dups:issues.append('duplicate_keys')
 if list(obj)!=j['fields']:issues.append('keys_or_order')
 for k,v in obj.items():
  if v is not None and (type(v) not in [str,int] if k==j['numeric_field'] else not isinstance(v,str)):issues.append('value_type')
 if truncated:issues.append('truncated')
 norm=lambda v:v.strip() if isinstance(v,str) else v
 target=norm(obj.get(j['target_field']));struct=not issues;encoding=any(isinstance(v,str) and v.strip()=='null' for v in obj.values());protocol=struct and not encoding
 if not struct:cat='invalid_structure'
 elif target is None:cat='json_null'
 elif target=='null':cat='string_null'
 elif target==j['source_value']:cat='source_value'
 elif target==j['control_value']:cat='control_value'
 elif target==j['supported_value']:cat='supported_value'
 elif isinstance(target,str) and target:cat='other_string'
 else:cat='empty_string'
 def correct(k):
  v=norm(obj.get(k));g=j['gold'][k]
  return protocol and k in obj and ((type(v)==int and v==g or isinstance(v,str) and v==str(g)) if type(g)==int else v==g)
 return dict(parsed_json=obj,target_value=target,target_category=cat,format_valid=struct,protocol_valid=protocol,issues=issues,string_null=cat=='string_null',source_misuse=j['evidence']!='supported' and cat=='source_value',control_misuse=j['evidence']!='supported' and cat=='control_value',other_string=cat in ['supported_value','other_string','empty_string'],true_null=cat=='json_null',target_correct=correct(j['target_field']),all_fields_correct=all(correct(k) for k in j['fields']),auxiliary_correct=correct(j['fields'][1]) if len(j['fields'])>1 else None)

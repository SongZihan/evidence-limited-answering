# JSON null 与字符串 null 分开计分。 / Distinguish JSON null from its string form.
"""Distinguish relationship misattribution from null encoding and parsing failures."""
import json
def score(j,raw,truncated=False):
 issues=[];dup=[];parsed=None
 def pairs(ps):
  obj={}
  for k,v in ps:
   if k in obj:dup.append(k)
   obj[k]=v
  return obj
 def bad(s):raise ValueError(s)
 try:parsed=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
 except ValueError:issues.append('invalid_json')
 obj=parsed if isinstance(parsed,dict) else {}
 if isinstance(parsed,dict):
  if dup:issues.append('duplicate_keys')
  if any(k not in obj for k in j['fields']):issues.append('missing_keys')
  if any(k not in j['fields'] for k in obj):issues.append('extra_keys')
  if set(obj)==set(j['fields']) and list(obj)!=j['fields']:issues.append('wrong_key_order')
  for k,v in obj.items():
   if v is not None and (type(v) not in [str,int] if k=='成立年份' else not isinstance(v,str)):issues.append('invalid_value_type')
 elif 'invalid_json' not in issues:issues.append('not_object')
 if truncated:issues.append('truncated')
 issues=sorted(set(issues));structural=not issues
 norm=lambda v:v.strip() if isinstance(v,str) else v
 target=norm(obj.get('总部城市'));present='总部城市' in obj
 stringnull=present and isinstance(target,str) and target=='null'
 violations=['string_null'] if any(isinstance(v,str) and norm(v)=='null' for v in obj.values()) else []
 strict=structural and not violations
 if not present:category='missing_or_invalid'
 elif target is None:category='json_null'
 elif stringnull:category='string_null'
 elif target==j['office_city']:category='office_city'
 elif target==j['launch_city']:category='launch_city'
 elif isinstance(target,str) and target:category='other_string'
 else:category='invalid_value'
 absent=j['evidence_version']!='supported'
 field_correct={}
 for k in j['fields']:
  val=norm(obj.get(k));gold=j['gold'][k]
  match=(type(val)==int and val==gold or isinstance(val,str) and val==str(gold)) if type(gold)==int else val==gold
  field_correct[k]=strict and k in obj and match
 aux=j['auxiliary_field']
 return dict(parsed_json=parsed,format_valid=structural,format_failure=not structural,
  format_status='valid_structure' if structural else '|'.join(issues),format_issues=issues,
  protocol_valid=strict,protocol_failure=not strict,encoding_violations=violations,
  target_present=present,target_value=target,gold_value=j['gold']['总部城市'],target_category=category,
  office_city_misuse=(category=='office_city') if absent else None,
  launch_city_misuse=(category=='launch_city') if absent else None,
  any_material_city_misuse=(category in ['office_city','launch_city']) if absent else None,
  other_string=(category=='other_string'),string_null=stringnull,
  nonnull_output=(present and target is not None) if absent else None,
  reliable_null=(strict and present and target is None) if absent else None,
  supported_correct=field_correct['总部城市'] if not absent else None,
  false_null=(present and target is None) if not absent else None,
  auxiliary_correct=field_correct[aux] if aux else None,field_correct=field_correct)

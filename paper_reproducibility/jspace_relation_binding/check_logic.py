"""Focused null-encoding, attribution and template-bootstrap checks."""
import json
from pathlib import Path
from scoring import score
from analyze import analyze
root=Path(__file__).resolve().parent
jobs=json.loads((root/'data/jobs.json').read_text(encoding='utf-8'))
j=next(j for j in jobs if j['condition']=='B' and j['relation_form']=='weak')
obj={k:j['gold'][k] for k in j['fields']}
assert score(j,json.dumps(obj))['reliable_null']
obj['总部城市']='null';r=score(j,json.dumps(obj))
assert r['string_null'] and not r['office_city_misuse'] and r['protocol_failure'] and not r['format_failure']
obj['总部城市']=j['office_city'];assert score(j,json.dumps(obj))['office_city_misuse']
obj['总部城市']=j['launch_city'];assert score(j,json.dumps(obj))['launch_city_misuse']
assert score(j,'{}')['format_failure'] and not score(j,'{}')['reliable_null']
assert score(j,json.dumps(dict(reversed(list(obj.items())))))['format_failure']
rows=[]
for x in jobs:
 if x['split']!='development':continue
 obj={k:x['gold'][k] for k in x['fields']}
 if x['relation_form']=='weak' and x['condition']=='D':obj['总部城市']=x['office_city']
 rows.append({**x,**score(x,json.dumps(obj))})
s=analyze(rows,json.loads((root/'protocol.json').read_text(encoding='utf-8')))
assert s['continuation']['primary']['estimate']==1 and s['continuation']['primary']['ci95']==[1,1]
assert s['continuation']['primary']['n_templates']==6 and s['continuation']['local_gate']
print('Null encoding, city attribution and template-cluster contrast checks passed.')

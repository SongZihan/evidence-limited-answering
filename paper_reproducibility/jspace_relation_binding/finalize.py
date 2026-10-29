"""Audit frozen behavior, cluster split and subsequent finite internal stage."""
import hashlib
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
cfg=read('protocol.json');jobs={j['job_id']:j for j in read('data/jobs.json')}
for p,h in read('data/lock.json').items():assert sha(p)==h,p
seal=read('results/prevalidation_seal.json')
for p,k in [('run.py','runner_sha256'),('scoring.py','scoring_sha256'),('analyze.py','analysis_sha256'),('results/development.jsonl','development_outputs_sha256')]:assert sha(p)==seal[k],p
assert not(set(cfg['template_split']['development'])&set(cfg['template_split']['validation']))
counts={};failures={};encoding={}
for split,n in cfg['counts'].items():
 rows=[json.loads(x) for x in (ROOT/f'results/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
 assert len(rows)==n and len({r['job_id'] for r in rows})==n
 assert {r['job_id'] for r in rows}=={j['job_id'] for j in jobs.values() if j['split']==split}
 assert all(all(r[k]==v for k,v in jobs[r['job_id']].items()) for r in rows)
 if split=='validation':assert all(r['utc']>=seal['utc'] for r in rows)
 counts[split]=n;failures[split]=sum(r['format_failure'] for r in rows);encoding[split]=sum(r['string_null'] for r in rows)
assert read('results/environment_main.json')['model_revision']==read('results/environment_pilot.json')['model_revision']
gate=read('results/gate_decision.json')
if gate['pass']:
 lens=read('results/lens_summary.json');selection=read('results/lens_layer_selection.json')
 assert len(selection['selected_layers'])==2 and selection['validation_readouts_seen'] is False
 assert len((ROOT/'results/lens_readouts.jsonl').read_text(encoding='utf-8').splitlines())==lens['n_readouts']
 assert len((ROOT/'results/controlled_value_scores.jsonl').read_text(encoding='utf-8').splitlines())==lens['n_controlled_value_scores']
 assert read('results/lens_protocol.json')['utc']<=selection['utc']
for p in ['report.html','relation_forms.png','template_effects.png']:assert (ROOT/'results'/p).is_file()
audit=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),counts=counts,structure_failures=failures,string_null_counts=encoding,
 locked_rules_unchanged=True,all_rows_preserved=True,template_holdout_verified=True,lens_completed=gate['pass'],causal_interventions=False,
 hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'results').iterdir() if p.suffix in ['.jsonl','.csv','.html','.png']})
(ROOT/'results/completion_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'results/status.json').write_text(json.dumps(dict(stage='report_complete',utc=audit['utc'],counts=counts,behavior_gate_pass=gate['pass'],lens_completed=gate['pass'],causal_interventions=False),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))

"""Final structural and intervention integrity checks; no outcome selection."""
import json, hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
def read(p):return json.loads(p.read_text())
def lines(name):return [json.loads(s) for s in (R/name).read_text().splitlines()]
assert read(R/'status.json')['stage']=='complete'
jobs=read(ROOT/'data/jobs.json');natural=lines('natural.jsonl');replay={r['job_id']:r for r in lines('replay.jsonl')};coverage=lines('coverage.jsonl');patch=lines('interventions.jsonl');candidate=lines('candidates.jsonl')
assert len(natural)==384 and {r['job_id'] for r in jobs}=={r['job_id'] for r in natural}
assert len(coverage)==384 and len(lines('readouts.jsonl'))==(768+sum(r['available'] for r in coverage))*15
assert len(candidate)==1152
for j in jobs:
 cs=[r for r in candidate if r['job_id']==j['job_id']];assert abs(sum(r['candidate_set_probability'] for r in cs)-1)<1e-6
 other=next(x for x in jobs if x['job_id']==j['job_id'][:-1]+('D' if j['condition']=='B' else 'B'))
 assert j['material']==other['material']
for p,h in read(ROOT/'data/lock.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
for r in patch:
 if r['kind']=='self':assert r['generated_token_ids']==replay[r['job_id']]['generated_token_ids']
selection=read(R/'selection.json');extra={}
if selection['selected']:
 assert len(patch)==672
 for kind in ['forward','reverse','full','random']:assert sum(r['split']=='validation' and r['kind']==kind for r in patch)==48
 u=torch.load(R/'frozen_basis.pt',map_location='cpu',weights_only=True);ru=torch.load(R/'random_basis.pt',map_location='cpu',weights_only=True)
 assert torch.allclose(u.T@u,torch.eye(u.shape[1]),atol=2e-5)
 assert torch.allclose(ru.T@ru,torch.eye(u.shape[1]),atol=2e-5)
 assert hashlib.sha256((R/'frozen_basis.pt').read_bytes()).hexdigest()==selection['basis_sha256']
 extra=dict(basis_shape=list(u.shape),orthonormality_max_error=float((u.T@u-torch.eye(u.shape[1])).abs().max()))
else:assert not patch
audit=dict(passed=True,natural=len(natural),readouts=len(lines('readouts.jsonl')),candidates=len(candidate),interventions=len(patch),self_checks=sum(r['kind']=='self' for r in patch),**extra)
(R/'completion_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))

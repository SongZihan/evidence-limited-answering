from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'run.py').read_text(encoding='utf-8')
s=s.replace("assert not (OUT/'pairs.jsonl').exists(),'Refuse overwrite'", "assert KEY=='glm'\n existing=[json.loads(line) for line in (OUT/'pairs.jsonl').read_text(encoding='utf-8').splitlines()]\n done={p['pair_id'] for p in existing};assert len(done)==len(existing)\n mc=dict(mc,budgets=[512],replay_budgets=[512])\n save('short_resume_amendment.json',dict(reason='User requested immediate truncation instead of long runs',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),retained_pairs=len(done),remaining_pairs=len(allpairs)-len(done),max_new_tokens=512,retries=0,original_results_sha256=sha(OUT/'pairs.jsonl'),frozen_basis_sha256=sha(OUT/'frozen_basis.pt')))")
s=s.replace("save('environment.json'", "save('short_resume_environment.json'")
s=s.replace("run_hash=sha(R/'run.py')", "run_hash=sha(R/'resume_short.py')")
s=s.replace("(OUT/'pairs.jsonl').open('w'", "(OUT/'pairs.jsonl').open('a'")
s=s.replace("(OUT/'development_candidates.jsonl').open('w'", "(OUT/'short_resume_checks.jsonl').open('w'")
s=s.replace("j=p['B'];r=dict(pair_id=p['pair_id'],", "j=p['B'];r=dict(generation_regime='short_512_no_retry',pair_id=p['pair_id'],")
a=s.index(' for i,p in enumerate(dev):');b=s.index(' for i,p in enumerate(others):',a)
s=s[:a]+" selected=read(OUT/'selection.json')['selected'];assert selected is not None\n u=torch.load(OUT/'frozen_basis.pt',weights_only=True).float();random_u=torch.load(OUT/'random_basis.pt',weights_only=True).float()\n assert sha(OUT/'frozen_basis.pt')==read(OUT/'selection.json')['basis_sha256']\n assert all(p['pair_id'] in done for p in dev)\n others=[p for p in others if p['pair_id'] not in done]\n status('short_resume',retained=len(done),remaining=len(others),max_new_tokens=512)\n"+s[b:]
s=s.replace("OUT/'evaluation_states.pt'", "OUT/'short_resume_states.pt'")
s=s.replace("status('evaluation',completed=i+1,total=len(others))", "status('evaluation_short',completed=i+1,total=len(others),overall_completed=len(done)+i+1,overall_total=len(allpairs))")
(R/'resume_short.py').write_text(s,encoding='utf-8')

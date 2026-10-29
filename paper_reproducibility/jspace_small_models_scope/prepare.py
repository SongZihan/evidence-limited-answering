import json,shutil,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
for d in ['data','results','logs']:(R/d).mkdir(exist_ok=True)
for ds in ['company','four','five']:
 for suffix in ['jobs','config']:shutil.copyfile(R.parent/f'jspace_more_models/data/{ds}_{suffix}.json',R/f'data/{ds}_{suffix}.json')
models={
 'qwen15':dict(repo='Qwen/Qwen2.5-1.5B-Instruct',name='Qwen2.5-1.5B-Instruct',gpu=0,chat_kwargs={}),
 'gemma2':dict(repo='LLM-Research/gemma-2-2b-it',name='Gemma-2-2B-it',gpu=1,chat_kwargs={}),
 'qwen3':dict(repo='Qwen/Qwen2.5-3B-Instruct',name='Qwen2.5-3B-Instruct',gpu=0,chat_kwargs={}),
 'phi4mini':dict(repo='microsoft/Phi-4-mini-instruct',name='Phi-4-mini-instruct',gpu=1,chat_kwargs={}),
 'qwen4':dict(repo='Qwen/Qwen3-4B',name='Qwen3-4B',gpu=0,chat_kwargs={'enable_thinking':False}),
 'llama3':dict(repo='LLM-Research/Llama-3.2-3B-Instruct',name='Llama-3.2-3B-Instruct',gpu=1,chat_kwargs={})}
for k,c in models.items():c['path']='models/'+k
plan=dict(models=models,datasets=['company','four','five'],seed=20261019,max_new_tokens=128,batch_size=16,per_model_n=3516,total_planned_n=21096,runtime='./llm_knowledge/.venv/bin/python',download='ModelScope SDK snapshot_download only. Named instruction models, original unquantized checkpoints. Config/tokenizer probes before weights; save metadata and checkpoint hashes. No model choice based on task output.',settings='BF16, eager, greedy, native chat, no extra system prompts, fixed original user prompts and gold. Qwen3 native nonthinking. Hard128-token cap, no truncation retries, no outcome-dependent prompt changes. Batch16 throughout. Chinese input retained for every model; poor language/format capability may invalidate replication inference.',scope='Behavioral universality test: full A/B/C/D across ten scenarios, expression and order as original dataset. No new interventions or J-lens fitting in this stage. Historical3516 prompt positions reused,852distinctmaterials, not a fresh independent corpus.',analysis='Strict JSON primary; separately strip exactly one complete outer Markdown fence for auxiliary content. Same existing company and nine-scene criteria; distinguish full, partial, no qualifying effect, insufficient output quality. Source errors, control errors, all target errors, encoding, format failures, supported diagnostic correctness reported together. Existing template bootstrap only, no extra significance fishing.',limits=['Six selected small checkpoints are a sample, not proof of universal behavior.','Three Qwen checkpoints share family; do not count as three independent families.','Llama3.2 Chinese is not a listed officially supported language; poor output quality is evidence of limited task suitability, not clean absence of effect.','Historical model runs differ in batch/budget; main comparisons are within each new model.','Negative and low-quality results retained, no post hoc removal.'])
(R/'protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
paths=[R/'protocol.json',R/'run.py']+[p for p in (R/'data').glob('*.json') if p.name!='lock.json']+list(R.glob('*_scoring.py'))+list(R.glob('*_analyze.py'))
(R/'data/lock.json').write_text(json.dumps({p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')

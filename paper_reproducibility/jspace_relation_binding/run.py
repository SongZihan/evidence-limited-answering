# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
"""One independent native-chat context per row; batched deterministic generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

ROOT=Path(__file__).resolve().parent

def write(path,obj):
    p=ROOT/path; tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def status(stage,**kw):
    obj=dict(stage=stage,utc=stamp());obj.update(kw)
    write('results/status.json',obj); print(json.dumps(obj,ensure_ascii=False),flush=True)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','main']);args=ap.parse_args()
    from scoring import score
    from analyze import analyze
    import torch
    import transformers
    config=json.loads((ROOT/'protocol.json').read_text())
    lock=json.loads((ROOT/'data/lock.json').read_text())
    for p,h in lock.items(): assert sha(ROOT/p)==h, f'Locked artifact changed: {p}'
    if args.stage=='main':
        pilot=json.loads((ROOT/'results/pilot_review.json').read_text())
        assert pilot['technical_pass'],pilot
    torch.set_num_threads(8);torch.manual_seed(config['seed']);random.seed(config['seed'])
    status('loading_model',mode=args.stage)
    tok=transformers.AutoTokenizer.from_pretrained(config['model_path'],local_files_only=True)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    model=transformers.AutoModelForCausalLM.from_pretrained(config['model_path'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation=config['attn_implementation']).cuda().eval()
    model.requires_grad_(False)
    for k in ['temperature','top_p','top_k']: setattr(model.generation_config,k,None)
    model_dir=Path(config['model_path'])
    # Full checkpoint digest, once, streams rather than reading tensors into RAM.
    hashes={}
    for p in sorted(model_dir.iterdir()):
        if p.suffix in ('.json','.jinja') or p.suffix=='.safetensors':
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            hashes[p.name]=h.hexdigest()
    revision=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    meta=dict(utc=stamp(),model_path=str(model_dir),model_revision=revision,
        revision_method='SHA256 of sorted JSON mapping of config/tokenizer/template/weight file SHA256 hashes',
        checkpoint_files=hashes,tokenizer_revision=hashes.get('tokenizer.json'),chat_template=tok.chat_template,
        precision='bfloat16',torch=torch.__version__,transformers=transformers.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),python=__import__('sys').version,
        seed=config['seed'],batch_size=config['batch_size'],max_new_tokens=config['max_new_tokens'],do_sample=False,
        eos_token_id=model.generation_config.eos_token_id,pad_token_id=tok.pad_token_id,
        code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},data_lock=lock)
    write('results/environment_'+args.stage+'.json',meta)
    jobs=json.loads((ROOT/'data/jobs.json').read_text())
    splits=['pilot'] if args.stage=='pilot' else ['development','validation']
    for split in splits:
        if split=='validation':
            # A concrete seal exists before any validation prompt reaches the model.
            write('results/prevalidation_seal.json',dict(utc=stamp(),data_lock=lock,
                development_outputs_sha256=sha(ROOT/'results/development.jsonl'),
                development_summary_sha256=sha(ROOT/'results/development_summary.json'),
                scoring_sha256=sha(ROOT/'scoring.py'),runner_sha256=sha(ROOT/'run.py'),
                analysis_sha256=sha(ROOT/'analyze.py'),decision='No prompt, data, scoring, thresholds or exclusions changed after development.'))
        selected=[j for j in jobs if j['split']==split]
        random.Random(config['seed']+splits.index(split)).shuffle(selected)
        outpath=ROOT/f'results/{split}.jsonl'
        # Resume finished rows without retrying invalid model responses.
        done=[]
        if outpath.exists():
            done=[json.loads(x) for x in outpath.read_text().splitlines() if x.strip()]
            assert len({r['job_id'] for r in done})==len(done)
        completed={r['job_id'] for r in done}
        todo=[j for j in selected if j['job_id'] not in completed]
        t0=time.monotonic()
        with outpath.open('a',encoding='utf-8',buffering=1) as handle:
            for start in range(0,len(todo),config['batch_size']):
                batch=todo[start:start+config['batch_size']]
                texts=[tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False) for j in batch]
                inputs=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    generated=model.generate(**inputs,do_sample=False,max_new_tokens=config['max_new_tokens'],
                        use_cache=True,pad_token_id=tok.pad_token_id,logits_to_keep=1)
                eos=model.generation_config.eos_token_id
                eos=set(eos if isinstance(eos,list) else [eos])
                for i,j in enumerate(batch):
                    tokens=generated[i,inputs['input_ids'].shape[1]:].tolist()
                    stop=next((k for k,v in enumerate(tokens) if v in eos),len(tokens))
                    truncated=stop==len(tokens) and len(tokens)>=config['max_new_tokens']
                    raw=tok.decode(tokens[:stop],skip_special_tokens=False)
                    row={**j,**score(j,raw,truncated), 'raw_output':raw,'prompt_text':texts[i],
                        'input_tokens':int(inputs['attention_mask'][i].sum()),
                        'p0_token_index':int(inputs['attention_mask'][i].sum())-1,
                        'generated_token_ids':tokens[:stop+1],'generated_tokens':min(stop+1,len(tokens)),
                        'model_revision':revision,'tokenizer_revision':meta['tokenizer_revision'],
                        'chat_template_ref':'environment_'+args.stage+'.json#chat_template',
                        'precision':config['precision'],'seed':config['seed'],'utc':stamp()}
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n');done.append(row)
                handle.flush();os.fsync(handle.fileno())
                status('running_'+split,completed=len(done),total=len(selected),elapsed_seconds=round(time.monotonic()-t0,1))
        assert len(done)==len(selected)
        summary=analyze(done,config)
        write(f'results/{split}_summary.json',summary)
        if split=='pilot':
            review=dict(utc=stamp(),n=len(done),technical_pass=all(r['format_valid'] for r in done),
                valid=sum(r['format_valid'] for r in done),failures=[dict(job_id=r['job_id'],raw=r['raw_output'],issues=r['format_issues']) for r in done if not r['format_valid']],
                policy='Behavioral correctness did not determine prompt changes or pilot acceptance.')
            write('results/pilot_review.json',review)
            status('pilot_complete',**review)
        if split=='development':
            write('results/development_review.json',dict(utc=stamp(),n=len(done),
                completed_unique=len({r['job_id'] for r in done}),format_failures=sum(r['format_failure'] for r in done),
                action='Keep all rows and locked rules; proceed to independent validation.'))
    if args.stage=='main':
        status('behavior_complete',gate=summary['continuation'])

if __name__=='__main__':
    try: main()
    except Exception as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise

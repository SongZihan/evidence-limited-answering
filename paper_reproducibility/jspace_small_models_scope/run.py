# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
import os,sys,json,time,hashlib,re,importlib,inspect,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;KEY=sys.argv[1]
def read(p):return json.loads((R/p).read_text(encoding='utf-8'))
def save(p,x):
 q=R/p;t=q.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(q)
def status(stage,**kw):
 x=dict(stage=stage,model=KEY,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);save(f'results/{KEY}_status.json',x);print(json.dumps(x),flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 import torch,transformers
 plan=read('protocol.json');cfg=plan['models'][KEY]
 for p,h in read('data/lock.json').items():assert sha(R/p)==h,p
 torch.set_num_threads(8);torch.manual_seed(plan['seed']);status('loading')
 tok=transformers.AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True);tok.padding_side='left'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=transformers.AutoModelForCausalLM.from_pretrained(cfg['path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
 params=sum(p.numel() for p in model.parameters());assert params<8_000_000_000,params
 for k in ['temperature','top_k','top_p']:setattr(model.generation_config,k,None)
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id);eos.discard(None)
 hashes={p.name:sha(p) for p in sorted(Path(cfg['path']).iterdir()) if p.suffix in ['.json','.safetensors','.model','.tiktoken','.jinja','.txt']}
 save(f'results/{KEY}_environment.json',dict(model=cfg,parameters=params,checkpoint_files=hashes,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),eos={i:tok.convert_ids_to_tokens(i) for i in sorted(eos)},chat_template=tok.chat_template,settings=plan,run_sha256=sha(R/'run.py')))
 extra={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
 def generate(batch,ds):
  texts=[tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],tokenize=False,add_generation_prompt=True,**cfg['chat_kwargs']) for j in batch]
  inp=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
  with torch.inference_mode():out=model.generate(**inp,do_sample=False,max_new_tokens=plan['max_new_tokens'],use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),**extra)
  scorer=importlib.import_module(ds+'_scoring').score;rows=[]
  for i,j in enumerate(batch):
   ids=out[i,inp['input_ids'].shape[1]:].tolist();stop=next((n for n,t in enumerate(ids) if t in eos),len(ids));truncated=stop==len(ids)
   raw=tok.decode(ids[:stop],skip_special_tokens=False);body=raw
   # Qwen3 nonthinking may emit its native empty think block. Preserve it in raw.
   if KEY=='qwen4':
    m=re.match(r'^\s*<think>\s*</think>\s*',body)
    if m:body=body[m.end():]
   m=re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n\s*```\s*',body,re.S|re.I);content=m[1] if m else body
   rows.append(dict(**j,**scorer(j,body,truncated),raw_output=body,generated_output=raw,prompt_text=texts[i],generated_token_ids=ids[:stop+1],input_tokens=int(inp['attention_mask'][i].sum()),truncated_generation=truncated,final_answer_found=True,outer_fence_removed=bool(m),content_score=scorer(j,content,truncated)))
  return rows
 smoke=[]
 for ds in plan['datasets']:smoke+=generate(read(f'data/{ds}_jobs.json')[:2],ds)
 save(f'results/{KEY}_smoke.json',smoke);status('smoke_complete',truncated=sum(r['truncated_generation'] for r in smoke))
 # Invalid/truncated smoke answers are task outcomes, not a reason to drop a model.
 for ds in plan['datasets']:
  jobs=read(f'data/{ds}_jobs.json');cfgdata=read(f'data/{ds}_config.json');path=R/f'results/{KEY}_{ds}.jsonl';assert not path.exists(),'Refuse overwrite';rows=[];start=time.monotonic()
  with path.open('w',encoding='utf-8',buffering=1) as f:
   for n in range(0,len(jobs),plan['batch_size']):
    rr=generate(jobs[n:n+plan['batch_size']],ds)
    for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    rows.extend(rr);f.flush();os.fsync(f.fileno());status('running',dataset=ds,completed=len(rows),total=len(jobs),elapsed_seconds=round(time.monotonic()-start,1))
  analyze=importlib.import_module(ds+'_analyze').analyze
  for mode in ['strict','content']:
   rr=rows if mode=='strict' else [{**r,**r['content_score']} for r in rows]
   s={split:analyze([r for r in rr if r['split']==split],cfgdata) for split in ['development','validation']} if ds=='company' else analyze(rr,cfgdata)
   save(f'results/{KEY}_{ds}_{mode}_summary.json',s)
 status('complete',n=3516);save(f'results/{KEY}_complete.json',dict(complete=True,n=3516))
if __name__=='__main__':
 try:main()
 except Exception:status('failed',error=traceback.format_exc());raise

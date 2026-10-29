# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
import os,sys,json,time,hashlib,re,importlib,traceback,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KEY=sys.argv[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def save(p,x):
 p=ROOT/p;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def status(stage,**kw):
 x=dict(stage=stage,model=KEY,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);save(f'results/{KEY}_status.json',x);print(json.dumps(x),flush=True)
def answer(raw):
 if KEY=='muse30b':
  # Prompt ends before assistant recipient; preserve full generated ATEM separately.
  parts=re.split(r'<\|eom\|><\|start\|>assistant',raw)
  finals=[p.split('<|message|>',1)[1] for p in parts if p.lstrip().startswith('to=user<|message|>')]
  return (finals[-1],True) if len(finals)==1 else ('',False)
 if KEY=='glm9b':
  if '<answer>' in raw:
   before,body=raw.split('<answer>',1)
   if '<answer>' in body:return '',False
   return body.removesuffix('</answer>'),True
  if '</think>' in raw:return raw.split('</think>',1)[1],True
  if '<think>' in raw:return '',False
 return raw,True
def main():
 import torch,transformers
 plan=read('protocol.json');cfg=plan['models'][KEY]
 for p,h in read('data/lock.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 torch.set_num_threads(8);torch.manual_seed(20261015);status('loading')
 custom=cfg['loader']=='causal_custom'
 tok=transformers.AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True,trust_remote_code=custom);tok.padding_side='left'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 loader=transformers.AutoModelForCausalLM if custom else transformers.AutoModelForImageTextToText
 model=loader.from_pretrained(cfg['path'],local_files_only=True,trust_remote_code=custom,torch_dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
 for k in ['temperature','top_k','top_p']:setattr(model.generation_config,k,None)
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id);eos.discard(None)
 special={i:tok.convert_ids_to_tokens(i) for i in sorted(eos)}
 status('checkpoint_hashing',eos=special)
 hashes={}
 for p in sorted(Path(cfg['path']).iterdir()):
  if p.suffix not in ['.json','.jinja','.safetensors','.py']:continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
  hashes[p.name]=h.hexdigest()
 revision=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
 save(f'results/{KEY}_environment.json',dict(model_path=cfg['path'],parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),chat_template=tok.chat_template,eos=special,checkpoint_files=hashes,revision=revision,settings=cfg))
 extra={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
 def generate(jobs,ds):
  texts=[tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],tokenize=False,add_generation_prompt=True,**cfg['chat_kwargs']) for j in jobs]
  inp=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
  with torch.inference_mode():out=model.generate(**inp,do_sample=False,max_new_tokens=cfg['max_new_tokens'],use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),**extra)
  score=importlib.import_module(ds+'_scoring').score;rows=[]
  for i,j in enumerate(jobs):
   ids=out[i,inp['input_ids'].shape[1]:].tolist();stop=next((k for k,v in enumerate(ids) if v in eos),len(ids));truncated=stop==len(ids) and len(ids)>=cfg['max_new_tokens']
   raw=tok.decode(ids[:stop],skip_special_tokens=False);body,found=answer(raw)
   rows.append({**j,**score(j,body,truncated), 'raw_output':body,'generated_output':raw,'final_answer_found':found,'truncated_generation':truncated,'prompt_text':texts[i],'generated_token_ids':ids[:stop+1],'input_tokens':int(inp['attention_mask'][i].sum()),'model_revision':revision})
  return rows
 status('technical_smoke')
 smoke=[]
 for ds in ['company','four']:smoke+=generate(read(f'data/{ds}_jobs.json')[:4],ds)
 save(f'results/{KEY}_smoke.json',smoke)
 assert all(r['final_answer_found'] and not r['truncated_generation'] for r in smoke),'Technical smoke: incomplete native answer; inspect output before formal run'
 if '--smoke-only' in sys.argv:status('smoke_complete');return
 for ds in plan['datasets']:
  jobs=read(f'data/{ds}_jobs.json');data_cfg=read(f'data/{ds}_config.json');torch.manual_seed(data_cfg['seed']);rows=[];path=ROOT/f'results/{KEY}_{ds}.jsonl'
  assert not path.exists(),'Avoid overwriting existing formal run'
  starttime=time.monotonic()
  with path.open('w',encoding='utf-8',buffering=1) as f:
   for start in range(0,len(jobs),cfg['batch']):
    rr=generate(jobs[start:start+cfg['batch']],ds)
    for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    rows+=rr;f.flush();os.fsync(f.fileno());status('running',dataset=ds,completed=len(rows),total=len(jobs),elapsed=round(time.monotonic()-starttime,1),truncated=sum(r['truncated_generation'] for r in rows))
  analyze=importlib.import_module(ds+'_analyze').analyze
  summary={s:analyze([r for r in rows if r['split']==s],data_cfg) for s in ['development','validation']} if ds=='company' else analyze(rows,data_cfg)
  save(f'results/{KEY}_{ds}_strict_summary.json',summary)
 save(f'results/{KEY}_complete.json',dict(complete=True,n=3516));status('complete')
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise

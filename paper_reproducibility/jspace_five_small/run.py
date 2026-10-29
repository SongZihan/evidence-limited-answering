# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
import json,hashlib,random,time,traceback,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(p,x):
 path=ROOT/p;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def status(stage,**kw):
 x=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);write('results/status.json',x);print(json.dumps(x),flush=True)
def read(p):return json.loads((ROOT/p).read_text())
def main():
 import torch,transformers
 from scoring import score
 from analyze import analyze
 cfg=read('protocol.json');lock=read('data/lock.json')
 for p,h in lock.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 assert not (ROOT/'results/outputs.jsonl').exists(),'Refuse accidental overwrite'
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);status('loading')
 tok=transformers.AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
 model=transformers.AutoModelForImageTextToText.from_pretrained(cfg['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval();model.requires_grad_(False)
 for k in ['temperature','top_k','top_p']:setattr(model.generation_config,k,None)
 hashes={}
 for path in sorted(Path(cfg['model_path']).iterdir()):
  if path.suffix not in ['.json','.jinja','.safetensors']:continue
  h=hashlib.sha256()
  with path.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  hashes[path.name]=h.hexdigest()
 prior=dict(model_revision=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
 write('results/checkpoint_files.json',hashes)
 write('results/environment.json',dict(model_revision=prior['model_revision'],checkpoint_verified=True,checkpoint_method="Full local checkpoint hashes; separate model from 8B",model_path=cfg["model_path"],parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),chat_template=tok.chat_template,precision='bfloat16',batch_size=cfg['batch_size'],lock=lock,seed=cfg['seed']))
 jobs=read('data/jobs.json');random.Random(cfg['seed']).shuffle(jobs);out=[];t0=time.monotonic()
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
 with (ROOT/'results/outputs.jsonl').open('w',encoding='utf-8',buffering=1) as f:
  for start in range(0,len(jobs),cfg['batch_size']):
   batch=jobs[start:start+cfg['batch_size']]
   texts=[tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for j in batch]
   inputs=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
   with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,max_new_tokens=cfg['max_new_tokens'],use_cache=True,pad_token_id=tok.pad_token_id,logits_to_keep=1)
   for i,j in enumerate(batch):
    ids=generated[i,inputs['input_ids'].shape[1]:].tolist();stop=next((k for k,v in enumerate(ids) if v in eos),len(ids));trunc=stop==len(ids) and len(ids)>=cfg['max_new_tokens'];raw=tok.decode(ids[:stop],skip_special_tokens=False)
    row={**j,**score(j,raw,trunc), 'raw_output':raw,'prompt_text':texts[i],'input_tokens':int(inputs['attention_mask'][i].sum()),'generated_token_ids':ids[:stop+1],'model_revision':prior['model_revision']}
    f.write(json.dumps(row,ensure_ascii=False)+'\n');out.append(row)
   f.flush();os.fsync(f.fileno());status('running',completed=len(out),total=len(jobs),elapsed=round(time.monotonic()-t0,1))
 write('results/summary.json',analyze(out,cfg));status('complete',n=len(out))
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise

# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
import json,hashlib,time,sys,os,importlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text())
def save(p,x):
 path=ROOT/p;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def status(stage,**kw):
 x=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);save('results/status.json',x);print(json.dumps(x),flush=True)
def main():
 import torch,transformers
 key=sys.argv[1];plan=read('protocol.json');datasets=next(x['datasets'] for x in plan['schedule'] if x['model']==key)
 for p,h in read('data/lock.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 torch.set_num_threads(8);torch.manual_seed(20261014);path=plan['models'][key];status('loading',model=key)
 tok=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True);tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token
 model=transformers.AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval();model.requires_grad_(False)
 for k in ['temperature','top_k','top_p']:setattr(model.generation_config,k,None)
 turn=tok.convert_tokens_to_ids('<turn|>');assert turn in tok.all_special_ids, (turn,tok.special_tokens_map)
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.update([turn,tok.eos_token_id]);eos.discard(None)
 hashes={};status('hashing_checkpoint',model=key)
 for p in sorted(Path(path).iterdir()):
  if p.suffix not in ['.json','.jinja','.safetensors']:continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  hashes[p.name]=h.hexdigest()
 revision=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
 save(f'results/{key}_environment.json',dict(model_path=path,model_revision=revision,checkpoint_files=hashes,parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),chat_template=tok.chat_template,eos_ids=sorted(eos),native_turn_end=turn,precision='bfloat16',max_new_tokens=128))
 for dataset in datasets:
  cfg=read(f'data/{dataset}_config.json');torch.manual_seed(cfg['seed']);jobs=read(f'data/{dataset}_jobs.json');score=importlib.import_module(dataset+'_scoring').score;analyze=importlib.import_module(dataset+'_analyze').analyze
  outpath=ROOT/f'results/{key}_{dataset}.jsonl';assert not outpath.exists(),'Avoid duplicate overwrite';rows=[];batchsize=cfg['batch_size'];starttime=time.monotonic()
  with outpath.open('w',encoding='utf-8',buffering=1) as f:
   for start in range(0,len(jobs),batchsize):
    batch=jobs[start:start+batchsize];texts=[tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False) for j in batch]
    inputs=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
    with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,max_new_tokens=128,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),logits_to_keep=1)
    for i,j in enumerate(batch):
     ids=generated[i,inputs['input_ids'].shape[1]:].tolist();stop=next((k for k,v in enumerate(ids) if v in eos),len(ids));truncated=stop==len(ids) and len(ids)>=128;raw=tok.decode(ids[:stop],skip_special_tokens=False)
     r={**j,**score(j,raw,truncated),'raw_output':raw,'prompt_text':texts[i],'generated_token_ids':ids[:stop+1],'model_revision':revision,'input_tokens':int(inputs['attention_mask'][i].sum()),'native_turn_boundary':True}
     rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
    f.flush();os.fsync(f.fileno());status('running',model=key,dataset=dataset,completed=len(rows),total=len(jobs),elapsed=round(time.monotonic()-starttime,1))
  if dataset=='company':
   for split in ['development','validation']:save(f'results/{key}_company_{split}_summary.json',analyze([r for r in rows if r['split']==split],cfg))
  else:save(f'results/{key}_{dataset}_summary.json',analyze(rows,cfg))
  status('dataset_complete',model=key,dataset=dataset,n=len(rows))
 save(f'results/{key}_complete.json',dict(complete=True,datasets=datasets));status('model_complete',model=key)
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',model=sys.argv[1],error=repr(e),traceback=traceback.format_exc());raise

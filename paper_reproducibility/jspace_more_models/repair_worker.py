import sys,os,json,time,importlib,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORKER=int(sys.argv[1])
def save(name,x):
 p=ROOT/'results'/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def status(stage,**kw):save(f'repair_worker_{WORKER}_status.json',dict(stage=stage,worker=WORKER,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
def main():
 import torch,transformers,run
 run.KEY='glm9b'
 tasks=json.loads((ROOT/'results/repair_tasks.json').read_text(encoding='utf-8'))[WORKER::2]
 output=ROOT/f'results/repair_worker_{WORKER}.jsonl'
 assert not output.exists(),'Do not overwrite attempts'
 if not tasks:output.touch();status('complete',total=0,completed=0);return
 cfg=json.loads((ROOT/'protocol.json').read_text())['models']['glm9b'];torch.set_num_threads(8);torch.manual_seed(20261015);status('loading',total=len(tasks))
 tok=transformers.AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True);tok.padding_side='left'
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=transformers.AutoModelForImageTextToText.from_pretrained(cfg['path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
 for k in ['temperature','top_k','top_p']:setattr(model.generation_config,k,None)
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id);eos.discard(None)
 completed=0;starttime=time.monotonic();pending=tasks
 with output.open('w',encoding='utf-8',buffering=1) as f:
  for budget in [8192,16384]:
   again=[]
   for start in range(0,len(pending),8):
    batch=pending[start:start+8];texts=[tok.apply_chat_template([dict(role='user',content=x['original']['user_prompt'])],tokenize=False,add_generation_prompt=True,**cfg['chat_kwargs']) for x in batch]
    assert all(text==x['original']['prompt_text'] for text,x in zip(texts,batch))
    inp=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
    status('running',budget=budget,completed=completed,total=len(tasks),current_batch=[x['original']['job_id'] for x in batch],elapsed=round(time.monotonic()-starttime,1))
    with torch.inference_mode():out=model.generate(**inp,do_sample=False,max_new_tokens=budget,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),logits_to_keep=1)
    for i,x in enumerate(batch):
     old=x['original'];ds=x['dataset'];ids=out[i,inp['input_ids'].shape[1]:].tolist();stop=next((k for k,v in enumerate(ids) if v in eos),len(ids));truncated=stop==len(ids) and len(ids)>=budget
     raw=tok.decode(ids[:stop],skip_special_tokens=False);body,found=run.answer(raw);sc=importlib.import_module(ds+'_scoring').score(old,body,truncated)
     record={**old,**sc,'raw_output':body,'generated_output':raw,'final_answer_found':found,'truncated_generation':truncated,'prompt_text':texts[i],'generated_token_ids':ids[:stop+1],'input_tokens':int(inp['attention_mask'][i].sum()),'repair':dict(max_new_tokens=budget,batch_size=len(batch),worker=WORKER,original_prefix_identical=ids[:len(old['generated_token_ids'])]==old['generated_token_ids'],original_record_sha256=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest())}
     f.write(json.dumps(dict(dataset=ds,record=record),ensure_ascii=False)+'\n')
     if truncated:again.append(x)
     else:completed+=1
    f.flush();os.fsync(f.fileno())
   pending=again
   if not pending:break
 status('complete',total=len(tasks),completed=completed,remaining_truncated=len(pending),elapsed=round(time.monotonic()-starttime,1))
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise

# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
import json,hashlib,re,time,traceback,sys
from pathlib import Path
import torch,transformers
ROOT=Path(__file__).resolve().parent;WORKER=int(sys.argv[1])
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 q=ROOT/p;t=q.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(q)
def status(stage,**kw):
 x=dict(stage=stage,worker=WORKER,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);save(f'results/status_{WORKER}.json',x);print(json.dumps(x),flush=True)
def score(j,text):
 target=j['target_field'];gold=j['gold'][target]
 try:
  obj=json.loads(text);valid=isinstance(obj,dict) and list(obj)==j['fields'] and all(v is None or type(v) in [str,int,float] for v in obj.values())
  value=obj.get(target,'MISSING') if isinstance(obj,dict) else 'INVALID'
  valid=valid and (value is None or isinstance(value,str) or (type(gold) in [int,float] and type(value) in [int,float]))
 except (ValueError,TypeError):obj={};value='INVALID';valid=False
 correct=valid and value==gold;source=bool(valid and not correct and value==j['source_value']);null=valid and value is None
 return dict(target=value,correct=correct,source_misuse=source,true_null=null,format_valid=valid,aux_correct=bool(valid and all(str(obj[k])==str(j['gold'][k]) for k in j['fields'] if k!=target)),category='format_failure' if not valid else 'correct' if correct else 'source' if source else 'string_null' if value=='null' else 'wrong_null' if null else 'control' if value==j.get('control_value') else 'other')
def main():
 cfg=read('protocol.json');pairs=[p for p in read('data/pairs.json') if p['worker']==WORKER];outpath=ROOT/f'results/pairs_{WORKER}.jsonl'
 assert not outpath.exists(),'Refuse overwrite'
 for p,h in read('data/lock.json').items():assert sha(ROOT/p)==h,p
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);status('loading')
 for name,h in read('data/prior_environment.json')['checkpoint_files'].items():assert sha(Path(cfg['model'])/name)==h,name
 u=torch.load(ROOT/'data/frozen_basis.pt',weights_only=True).float();random_u=torch.load(ROOT/'data/random_basis.pt',weights_only=True).float()
 assert u.shape==(4096,4) and torch.allclose(u.T@u,torch.eye(4),atol=2e-5)
 assert sha(ROOT/'data/frozen_basis.pt')==read('data/original_selection.json')['basis_sha256']
 tok=transformers.AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True);tok.pad_token=tok.eos_token
 model=transformers.AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).cuda().eval();model.requires_grad_(False)
 for k in ['temperature','top_p','top_k']:setattr(model.generation_config,k,None)
 save(f'results/environment_{WORKER}.json',dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),checkpoint_verified=True,basis_hash=sha(ROOT/'data/frozen_basis.pt'),random_hash=sha(ROOT/'data/random_basis.pt'),run_hash=sha(ROOT/'run.py')))
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id)
 def generate(ids,vector=None):
  calls=[];handle=None
  if vector is not None:
   def hook(module,args,res):
    h=res if torch.is_tensor(res) else res[0]
    if not calls:
     assert h.shape[1]==len(ids);h=h.clone();h[0,-1]=vector.to(h);calls.append(1)
     return h if torch.is_tensor(res) else (h,)+res[1:]
   # 仅在预定位置替换状态；后续沿用接收运行。 / Patch the selected position, then resume the recipient.
   handle=model.model.layers[33].register_forward_hook(hook)
  x=torch.tensor([ids],device='cuda')
  try:
   with torch.inference_mode():out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=128,use_cache=True,pad_token_id=tok.pad_token_id,logits_to_keep=1)[0,len(ids):].tolist()
  finally:
   if handle:handle.remove()
  if vector is not None:assert len(calls)==1
  return out
 states={}
 def context(j):
  prompt=tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False);pids=tok.encode(prompt,add_special_tokens=False);gen=generate(pids);text=tok.decode(gen,skip_special_tokens=True)
  natural=dict(uid=j['uid'],kind='natural',raw_output=text,generated_token_ids=gen,truncated=gen[-1] not in eos,**score(j,text))
  bounds=[]
  for n in range(1,len(gen)+1):
   prefix=tok.decode(gen[:n],skip_special_tokens=True)
   if re.search('"'+re.escape(j['target_field'])+r'"\s*:\s*$',prefix):bounds.append((n,prefix))
  if not bounds:return natural,None
  n,prefix=bounds[-1];ids=pids+gen[:n];captured=[]
  def hook(module,args,res):captured.append((res if torch.is_tensor(res) else res[0])[0,-1].detach().cpu().clone())
  handle=model.model.layers[33].register_forward_hook(hook)
  try:
   with torch.inference_mode():model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1)
  finally:handle.remove()
  assert len(captured)==1;vec=captured[0];states[j['uid']]=vec
  replay_ids=generate(ids);raw=prefix+tok.decode(replay_ids,skip_special_tokens=True)
  replay=dict(uid=j['uid'],kind='replay',raw_output=raw,generated_token_ids=replay_ids,truncated=replay_ids[-1] not in eos,natural_equal=gen[n:]==replay_ids,**score(j,raw))
  return natural,dict(job=j,ids=ids,prefix=prefix,vec=vec,replay=replay,prefix_tokens=n)
 def intervene(rec,don,kind,vec):
  ids=generate(rec['ids'],vec);raw=rec['prefix']+tok.decode(ids,skip_special_tokens=True)
  if kind.startswith('self'):assert ids==rec['replay']['generated_token_ids'],'Self replacement mismatch'
  delta=vec.float()-rec['vec'].float();realized=vec.to(torch.bfloat16).float()-rec['vec'].float()
  return dict(uid=rec['job']['uid'],donor_uid=don['job']['uid'],kind=kind,raw_output=raw,generated_token_ids=ids,truncated=ids[-1] not in eos,intended_delta_norm=float(delta.norm()),realized_delta_norm=float(realized.norm()),**score(rec['job'],raw))
 start=time.monotonic()
 with outpath.open('w',encoding='utf-8',buffering=1) as f:
  for i,p in enumerate(pairs):
   nb,b=context(p['B']);nd,d=context(p['D']);rows=[nb,nd];boundaries={}
   for c in [b,d]:
    if c:rows.append(c['replay']);boundaries[c['job']['condition']]=dict(prefix=c['prefix'],input_ids=c['ids'],prefix_tokens=c['prefix_tokens'])
   available=b is not None and d is not None
   if available:
    rows.append(intervene(b,b,'self_B',b['vec']));rows.append(intervene(d,d,'self_D',d['vec']))
    bh,dh=b['vec'].float(),d['vec'].float();delta=bh-dh;project=u@(u.T@delta)
    rows.append(intervene(d,b,'forward',dh+project));rows.append(intervene(b,d,'reverse',bh-project));rows.append(intervene(d,b,'full',bh))
    rd=random_u@(random_u.T@delta)
    if float(rd.norm())<1e-12:rd=random_u[:,0]
    randomvec=dh+rd*(project.norm()/rd.norm());rows.append(intervene(d,b,'random',randomvec))
   result=dict(pair_id=p['pair_id'],dataset=p['B']['dataset'],family=p['B']['family'],template_id=p['B']['template_id'],split=p['B']['split'],form=p['B']['form'],evidence=p['B']['evidence'],target_order=p['B']['target_order'],target_field=p['B']['target_field'],gold=p['B']['gold'][p['B']['target_field']],available=available,boundaries=boundaries,rows=rows)
   f.write(json.dumps(result,ensure_ascii=False)+'\n')
   status('running',completed=i+1,total=len(pairs),family=result['family'],elapsed_seconds=round(time.monotonic()-start,1))
 torch.save(states,ROOT/f'results/states_{WORKER}.pt');status('complete',completed=len(pairs),total=len(pairs),elapsed_seconds=round(time.monotonic()-start,1))
if __name__=='__main__':
 try:main()
 except Exception:status('failed',error=traceback.format_exc());raise

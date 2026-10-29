# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
import json,hashlib,re,time,traceback,sys,inspect
from pathlib import Path
import torch,transformers
R=Path(__file__).resolve().parent;KEY=sys.argv[1];OUT=R/'results'/KEY;START=time.time()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 p=OUT/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)
def status(stage,**kw):
 save('status.json',dict(stage=stage,elapsed_seconds=time.time()-START,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw));print(stage,kw,flush=True)
def score(j,text):
 def parse(s):
  try:
   o=json.loads(s);v=o.get(j['target_field'],'MISSING') if isinstance(o,dict) else 'INVALID';valid=isinstance(o,dict) and list(o)==j['fields'] and all(v is None or type(v) in [str,int,float] for v in o.values())
  except (ValueError,TypeError):o={};v='INVALID';valid=False
  good=valid and v==j['gold'][j['target_field']];source=valid and not good and v==j['source_value']
  return dict(target=v,correct=bool(good),source_misuse=bool(source),true_null=bool(valid and v is None),format_valid=bool(valid),category='format_failure' if not valid else 'correct' if good else 'source' if source else 'string_null' if v=='null' else 'wrong_null' if v is None else 'control' if v==j['control_value'] else 'other')
 strict=parse(text);m=re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n\s*```\s*',text,re.S|re.I)
 return dict(**parse(m[1] if m else text),strict=strict,outer_fence_removed=bool(m))
def main():
 cfg=read(R/'protocol.json');mc=cfg['models'][KEY];allpairs=read(R/'data/pairs.json');layer=mc['layer']
 for path,digest in read(R/'data/lock.json').items():assert sha(R/path)==digest,path
 assert transformers.__version__==mc['expected_transformers']
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);status('verify_checkpoint')
 oldroot=R.parent/mc['original_results']
 oldenv=read(oldroot/'environment.json') if KEY=='qwen2b' else read(R.parent/'jspace_relation_mechanism_round1/data/prior_environment.json')
 verified={}
 for name,digest in oldenv['checkpoint_files'].items():
  actual=sha(Path(mc['path'])/name);assert actual==digest,name;verified[name]=actual
 status('loading')
 tok=transformers.AutoTokenizer.from_pretrained(mc['path'],local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 loader=transformers.AutoModelForImageTextToText if KEY=='qwen2b' else transformers.AutoModelForCausalLM
 model=loader.from_pretrained(mc['path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
 tc=getattr(model.config,'text_config',model.config)
 found=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.ModuleList) and len(m)==tc.num_hidden_layers and n.endswith('layers')];assert len(found)==1
 layername,layers=found[0]
 for k in ['temperature','top_p','top_k']:setattr(model.generation_config,k,None)
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id);eos.discard(None)
 extra={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
 u=torch.load(R/'data'/f'{KEY}_frozen_basis.pt',map_location='cpu',weights_only=True).float();random_u=torch.load(R/'data'/f'{KEY}_random_basis.pt',map_location='cpu',weights_only=True).float()
 assert u.shape==(tc.hidden_size,mc['rank']) and random_u.shape==u.shape
 assert torch.allclose(u.T@u,torch.eye(mc['rank']),atol=1e-4)
 save('environment.json',dict(model=mc,layer_module=layername,torch=torch.__version__,transformers=transformers.__version__,checkpoint_files=verified,basis_sha256=sha(R/'data'/f'{KEY}_frozen_basis.pt'),random_basis_sha256=sha(R/'data'/f'{KEY}_random_basis.pt'),script_sha256=sha(R/'run.py'),eos=sorted(eos),gpu=torch.cuda.get_device_name(),chat_template=tok.chat_template))
 def generate(ids,vec=None):
  calls=[];handle=None
  if vec is not None:
   def hook(module,args,res):
    h=res if torch.is_tensor(res) else res[0]
    if not calls:
     assert h.shape[1]==len(ids);h=h.clone();h[0,-1]=vec.to(h);calls.append(1)
     return h if torch.is_tensor(res) else (h,)+res[1:]
   # 仅在预定位置替换状态；后续沿用接收运行。 / Patch the selected position, then resume the recipient.
   handle=layers[layer].register_forward_hook(hook)
  x=torch.tensor([ids],device='cuda')
  try:
   with torch.inference_mode():out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=cfg['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),**extra)[0,len(ids):].tolist()
  finally:
   if handle:handle.remove()
  if vec is not None:assert calls==[1]
  return out
 def row(j,kind,ids,prefix=''):
  raw=prefix+tok.decode(ids,skip_special_tokens=True);s=score(j,raw);tr=not ids or ids[-1] not in eos
  if tr:
   for q in [s,s['strict']]:q.update(correct=False,format_valid=False,category='truncated',true_null=False,source_misuse=False)
  return dict(uid=j['uid'],kind=kind,raw_output=raw,generated_token_ids=ids,truncated=tr,**s)
 def context(j,need):
  prompt=tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False);pids=tok.encode(prompt,add_special_tokens=False)
  ids=generate(pids);natural=row(j,'natural',ids);natural['prompt_text']=prompt
  if not need:return natural,None
  bounds=[]
  for n in range(1,len(ids)+1):
   prefix=tok.decode(ids[:n],skip_special_tokens=True)
   good=re.fullmatch(r'\s*\{\s*"总部城市"\s*:\s*',prefix) if KEY=='qwen8b' else re.search(r'"总部城市"\s*:\s*$',prefix)
   if good:bounds.append((n,prefix))
  if not bounds:return natural,None
  n,prefix=bounds[-1] if KEY=='qwen8b' else bounds[0];inputs=pids+ids[:n];state=[]
  def hook(module,args,res):state.append((res if torch.is_tensor(res) else res[0])[0,-1].detach().cpu().clone())
  h=layers[layer].register_forward_hook(hook)
  try:
   with torch.inference_mode():model(input_ids=torch.tensor([inputs],device='cuda'),use_cache=False,**extra)
  finally:h.remove()
  assert len(state)==1
  out=generate(inputs);base=row(j,'replay',out,prefix);base['natural_equal']=out==ids[n:]
  return natural,dict(j=j,ids=inputs,prefix=prefix,state=state[0],replay=base)
 def patch(c,don,kind,vec):
  ids=generate(c['ids'],vec);r=row(c['j'],kind,ids,c['prefix']);r['donor_uid']=don['j']['uid']
  if kind.startswith('self'):assert ids==c['replay']['generated_token_ids'],'Self replacement mismatch'
  r['delta_norm']=float((vec.float()-c['state'].float()).norm());return r
 def process(p):
  need=p['B']['form']=='weak';nb,b=context(p['B'],need);nd,d=context(p['D'],need);rows=[nb,nd];bounds={}
  for c in [b,d]:
   if c:rows.append(c['replay']);bounds[c['j']['condition']]=dict(input_ids=c['ids'],prefix=c['prefix'])
  if b and d:
   bh=b['state'].float();dh=d['state'].float();delta=bh-dh;proj=u@(u.T@delta);rand=random_u@(random_u.T@delta)
   if float(rand.norm())<1e-12:rand=random_u[:,0]
   rows.extend([patch(b,b,'self_B',bh),patch(d,d,'self_D',dh),patch(d,b,'forward',dh+proj),patch(b,d,'reverse',bh-proj),patch(d,b,'full',bh),patch(d,b,'random',dh+rand*(proj.norm()/rand.norm()))])
  j=p['B'];result=dict(pair_id=p['pair_id'],template_id=j['template_id'],group_id=j['group_id'],evidence=j['evidence'],form=j['form'],available=bool(b and d),intervention_planned=need,boundaries=bounds,rows=rows)
  return result,({j['uid']:c['state'] for c in [b,d] if c for j in [c['j']]})
 if '--smoke' in sys.argv:
  r,states=process(allpairs[0]);save('smoke.json',r);assert r['available'];status('smoke_complete');return
 done=set()
 if (OUT/'pairs.jsonl').exists():done={json.loads(l)['pair_id'] for l in (OUT/'pairs.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()}
 f=(OUT/'pairs.jsonl').open('a',encoding='utf-8',buffering=1);cache={};current=None
 for i,p in enumerate(allpairs):
  if p['pair_id'] in done:continue
  template=p['B']['template_id']
  if template!=current:
   if current:torch.save(cache,OUT/f'states_{current}.pt')
   cache={};current=template
  r,states=process(p);cache.update(states);f.write(json.dumps(r,ensure_ascii=False)+'\n');status('evaluation',completed=i+1,total=len(allpairs))
 if current:torch.save(cache,OUT/f'states_{current}.pt')
 f.close();status('complete',completed=len(allpairs),total=len(allpairs))
if __name__=='__main__':
 try:main()
 except Exception:status('failed',error=traceback.format_exc());raise

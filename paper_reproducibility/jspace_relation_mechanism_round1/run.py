# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
import json, hashlib, re, time, traceback, sys
from pathlib import Path
import torch, transformers
ROOT=Path(__file__).resolve().parent
OLD=Path('./jspace_task_selection')
sys.path.insert(0,str(OLD/'official/jacobian-lens-main'))
import jlens
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 q=ROOT/p;tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(q)
def status(stage,**kw):
 x=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);write('results/status.json',x);print(json.dumps(x),flush=True)
def score(j,text):
 try:
  obj=json.loads(text);value=obj.get('总部城市','MISSING') if isinstance(obj,dict) else 'INVALID'
  valid=isinstance(obj,dict) and list(obj)==j['fields'] and all(v is None or isinstance(v,(str,int,float)) for v in obj.values())
 except (ValueError,TypeError):obj={};value='INVALID';valid=False
 correct=valid and value==j['hq_city'];null=valid and value is None
 return dict(target=value,format_valid=valid,correct=correct,true_null=null,office_misuse=bool(valid and value==j['office_city']),auxiliary_correct=bool(valid and str(obj.get(j['fields'][1]))==str(j['gold'][j['fields'][1]])),category='format_failure' if not valid else 'correct' if correct else 'string_null' if value=='null' else 'office' if value==j['office_city'] else 'other')
def gate(rows):
 def rate(form,cond,template=None):
  rs=[r for r in rows if r['evidence']=='absent' and r['form']==form and r['condition']==cond and (template is None or r['template_id']==template)]
  return sum(r['office_misuse'] for r in rs)/len(rs)
 weak=rate('weak','D')-rate('weak','B');direct=rate('direct','D')-rate('direct','B')
 ts=sorted({r['template_id'] for r in rows});diff={t:rate('weak','D',t)-rate('weak','B',t) for t in ts}
 supported=[r for r in rows if r['evidence']!='absent'];accuracy=sum(r['correct'] for r in supported)/len(supported)
 return dict(passed=weak>=.125 and sum(v>0 for v in diff.values())>=3 and direct<weak and accuracy>=.9,weak_D_minus_B=weak,direct_D_minus_B=direct,interaction=weak-direct,template_differences=diff,supported_accuracy=accuracy)

def main():
 cfg=read('protocol.json');jobs=read('data/jobs.json')
 assert not (ROOT/'results/natural.jsonl').exists(),'Refuse overwrite'
 for p,h in read('data/lock.json').items():assert sha(ROOT/p)==h,p
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);status('provenance')
 previous=read('data/prior_environment.json')
 for name,h in previous['checkpoint_files'].items():assert sha(Path(cfg['model_path'])/name)==h,name
 meta=json.loads((OLD/'results/lens_metadata.json').read_text())
 assert sha(Path(cfg['model_path'])/'config.json')==meta['model_config_sha256']
 assert sha(OLD/'results/jacobian_lens.pt')==meta['lens_sha256']
 tok=transformers.AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True);tok.pad_token=tok.eos_token
 model=transformers.AutoModelForCausalLM.from_pretrained(cfg['model_path'],torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).cuda().eval();model.requires_grad_(False)
 for k in ['temperature','top_p','top_k']:setattr(model.generation_config,k,None)
 wrapped=jlens.from_hf(model,tok,force_bos=False)
 lens=jlens.JacobianLens.load(str(OLD/'results/jacobian_lens.pt'));lens.jacobians={l:j.cuda() for l,j in lens.jacobians.items() if l in cfg['layers']}
 tokens={s:tok.encode(s,add_special_tokens=False) for s in ['武汉','成都','上海','南京','null']}
 write('results/environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),checkpoint_verified=True,lens_metadata=meta,tokens=tokens,run_sha256=sha(ROOT/'run.py')))
 files={k:(ROOT/f'results/{k}.jsonl').open('w',encoding='utf-8',buffering=1) for k in ['natural','replay','readouts','candidates','coverage','interventions']}
 def emit(k,x):files[k].write(json.dumps(x,ensure_ascii=False)+'\n')
 def generate(ids,patch=None):
  calls=[];handle=None
  if patch:
   layer,vector=patch
   def hook(module,args,res):
    h=res if torch.is_tensor(res) else res[0]
    if not calls:
     assert h.shape[1]==len(ids);h=h.clone();h[0,-1]=vector.to(h);calls.append(1)
     return h if torch.is_tensor(res) else (h,)+res[1:]
   # 仅在预定位置替换状态；后续沿用接收运行。 / Patch the selected position, then resume the recipient.
   handle=model.model.layers[layer].register_forward_hook(hook)
  x=torch.tensor([ids],device='cuda')
  try:
   with torch.inference_mode():out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=128,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)[0,len(ids):].tolist()
  finally:
   if handle:handle.remove()
  if patch:assert len(calls)==1
  return out
 contexts={};natural=[];states={}
 def inspect(j,ids,position):
  captured={};handles=[]
  for l in cfg['layers']+[35]:
   def hook(module,args,res,l=l):captured[l]=(res if torch.is_tensor(res) else res[0])[0,-1].detach().clone()
   handles.append(model.model.layers[l].register_forward_hook(hook))
  try:
   with torch.inference_mode():model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1)
  finally:
   for h in handles:h.remove()
  with torch.inference_mode():
   for l,h in captured.items():
    for method in (['Logit Lens'] if l==35 else ['J-lens','Logit Lens']):
     vec=lens.transport(h[None].float(),l) if method=='J-lens' else h[None]
     logits=wrapped.unembed(vec).float()[0]
     emit('readouts',dict(job_id=j['job_id'],position=position,layer=l,method=method,scores={s:float(logits[v[0]]) for s,v in tokens.items() if len(v)==1}))
  if position=='PN':return {l:captured[l].cpu() for l in cfg['patch_layers']}
 def candidates(j,prompt):
  text=prompt+'{"总部城市":';prefix=tok.encode(text,add_special_tokens=False)
  values=[]
  for value in [None,j['city_a'],j['city_b']]:
   continuation=' '+json.dumps(value,ensure_ascii=False)+',';full=tok.encode(text+continuation,add_special_tokens=False)
   assert full[:len(prefix)]==prefix,'Unstable token boundary'
   x=torch.tensor([full],device='cuda')
   with torch.inference_mode():logits=model(input_ids=x,use_cache=False).logits[0,len(prefix)-1:-1].float().log_softmax(-1)
   target=torch.tensor(full[len(prefix):],device='cuda');lp=float(logits.gather(-1,target[:,None]).sum())
   values.append(dict(value=value,continuation=continuation,logp=lp))
  probs=torch.softmax(torch.tensor([v['logp'] for v in values]),0).tolist()
  for v,p in zip(values,probs):emit('candidates',dict(job_id=j['job_id'],position='P1',**v,candidate_set_probability=p))
 def process(split):
  subset=[j for j in jobs if j['split']==split]
  for i,j in enumerate(subset):
   prompt=tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False);pids=tok.encode(prompt,add_special_tokens=False)
   generated=generate(pids);raw=tok.decode(generated,skip_special_tokens=True)
   row=dict(**j,raw_output=raw,generated_token_ids=generated,prompt_text=prompt,input_tokens=len(pids),truncated=generated[-1]!=tok.eos_token_id,**score(j,raw));natural.append(row);emit('natural',row)
   inspect(j,pids,'P0');bounds=[]
   for n in range(1,len(generated)+1):
    prefix=tok.decode(generated[:n],skip_special_tokens=True)
    if re.fullmatch(r'\s*\{\s*"总部城市"\s*:\s*',prefix):bounds.append((n,prefix))
   emit('coverage',dict(job_id=j['job_id'],available=bool(bounds),boundary=bounds[-1][0] if bounds else None))
   if bounds:
    n,prefix=bounds[-1];ids=pids+generated[:n];captured=inspect(j,ids,'PN');states[j['job_id']]=captured
    replay=generate(ids);text=prefix+tok.decode(replay,skip_special_tokens=True)
    base=dict(job_id=j['job_id'],raw_output=text,generated_token_ids=replay,natural_equal=(generated[n:]==replay),**score(j,text));emit('replay',base)
    contexts[j['job_id']]=dict(job=j,ids=ids,prefix=prefix,states=captured,baseline=base)
   forced=tok.encode(prompt+'{"总部城市":',add_special_tokens=False);inspect(j,forced,'P1');candidates(j,prompt)
   status(split,completed=i+1,total=len(subset))
  torch.save(states,ROOT/'results/paired_states.pt')
  write('results/context_boundaries.json',{k:dict(input_ids=c['ids'],prefix=c['prefix']) for k,c in contexts.items()})
 def pairs(split):
  return [(contexts.get(j['job_id']),contexts.get(j['job_id'][:-1]+'D')) for j in jobs if j['split']==split and j['form']=='weak' and j['condition']=='B']
 def patched(rec,don,layer,kind,vec,rank=None):
  out=generate(rec['ids'],(layer,vec.cuda()));text=rec['prefix']+tok.decode(out,skip_special_tokens=True)
  r=dict(job_id=rec['job']['job_id'],donor_id=don['job']['job_id'],split=rec['job']['split'],template_id=rec['job']['template_id'],evidence=rec['job']['evidence'],kind=kind,layer=layer,rank=rank,raw_output=text,generated_token_ids=out,**score(rec['job'],text))
  if kind=='self':assert out==rec['baseline']['generated_token_ids'],'Self patch failed'
  emit('interventions',r);return r
 def project(rec,don,layer,u):
  a=rec['states'][layer].float();delta=don['states'][layer].float()-a;return a+u@(u.T@delta)
 process('development');decision=gate([r for r in natural if r['split']=='development']);write('results/gate_decision.json',decision)
 chosen=None;bases={}
 if decision['passed']:
  ps=pairs('development');assert len(ps)==48 and all(b and d for b,d in ps),'Incomplete development PN'
  for layer in cfg['patch_layers']:
   # 不减均值：保留两种指令的整体偏移。 / Uncentered SVD retains the instruction offset.
   matrix=torch.stack([b['states'][layer].float()-d['states'][layer].float() for b,d in ps]);_,sv,vh=torch.linalg.svd(matrix,full_matrices=False)
   for rank in cfg['ranks']:bases[(layer,rank)]=vh[:rank].T.contiguous()
   torch.save(dict(singular_values=sv,basis=vh[:4].T),ROOT/f'results/development_svd_L{layer}.pt')
  stats=[]
  for layer in cfg['patch_layers']:
   for b,d in ps:
    for c in [b,d]:patched(c,c,layer,'self',c['states'][layer])
   for rank in cfg['ranks']:
    rs=[patched(d,b,layer,'development_forward',project(d,b,layer,bases[(layer,rank)]),rank) for b,d in ps]
    gains={e:sum(r['correct']-d['baseline']['correct'] for r,(b,d) in zip(rs,ps) if r['evidence']==e) for e in ['absent','supported','swapped']}
    misuse_reduction=sum(d['baseline']['office_misuse']-r['office_misuse'] for r,(b,d) in zip(rs,ps) if r['evidence']=='absent')
    stats.append(dict(layer=layer,rank=rank,gains=gains,eligible=misuse_reduction>0 and gains['supported']>=-1 and gains['swapped']>=-1,macro_gain=sum(gains.values())/48))
  eligible=[s for s in stats if s['eligible']]
  chosen=sorted(eligible,key=lambda s:(-s['gains']['absent'],s['rank'],s['layer']))[0] if eligible else sorted(stats,key=lambda s:(-s['macro_gain'],s['rank'],s['layer']))[0]
  torch.save(bases[(chosen['layer'],chosen['rank'])],ROOT/'results/frozen_basis.pt')
  write('results/selection.json',dict(selected=chosen,candidates=stats,validation_seen=False,basis_sha256=sha(ROOT/'results/frozen_basis.pt')))
 else:write('results/selection.json',dict(selected=None,reason='Predefined development gate failed',validation_seen=False))
 status('selection_frozen',gate=decision,selected=chosen)
 process('validation')
 if chosen:
  layer,rank=chosen['layer'],chosen['rank'];u=bases[(layer,rank)];gen=torch.Generator().manual_seed(cfg['seed']);random_u=torch.linalg.qr(torch.randn(u.shape,generator=gen)).Q;torch.save(random_u,ROOT/'results/random_basis.pt')
  for b,d in pairs('validation'):
   if not b or not d:continue
   for c in [b,d]:patched(c,c,layer,'self',c['states'][layer])
   vec=project(d,b,layer,u);patched(d,b,layer,'forward',vec,rank)
   patched(b,d,layer,'reverse',project(b,d,layer,u),rank);patched(d,b,layer,'full',b['states'][layer],4096)
   a=d['states'][layer].float();delta=b['states'][layer].float()-a;rd=random_u@(random_u.T@delta);norm=torch.linalg.vector_norm(vec-a)
   if float(torch.linalg.vector_norm(rd))<1e-12:rd=random_u[:,0]
   random_vec=a+rd*(norm/torch.linalg.vector_norm(rd));patched(d,b,layer,'random',random_vec,rank)
 for f in files.values():f.close()
 status('complete',natural=len(natural),intervention_run=bool(chosen))
if __name__=='__main__':
 try:main()
 except Exception:
  status('failed',error=traceback.format_exc());raise

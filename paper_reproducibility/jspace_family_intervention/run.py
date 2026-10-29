# 同一事实配对运行；保留失败样本，避免按输出筛选。
# Matched facts, fixed decoding; failures stay in the planned denominator.
import json,hashlib,re,time,traceback,sys,inspect
from pathlib import Path
import torch,transformers
from native import answer_span,locate
R=Path(__file__).resolve().parent;KEY=sys.argv[1];OUT=R/'results'/KEY
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def status(stage,**kw):save('status.json',dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw));print(stage,kw,flush=True)
def score(j,text):
 def parse(t):
  try:
   o=json.loads(t);v=o.get(j['target_field'],'MISSING') if isinstance(o,dict) else 'INVALID';valid=isinstance(o,dict) and list(o)==j['fields'] and all(v is None or type(v) in [str,int,float] for v in o.values())
  except (ValueError,TypeError):o={};v='INVALID';valid=False
  good=valid and v==j['gold'][j['target_field']];source=valid and not good and v==j['source_value']
  return dict(target=v,correct=bool(good),source_misuse=bool(source),true_null=bool(valid and v is None),format_valid=valid,aux_correct=bool(valid and all(str(o[k])==str(j['gold'][k]) for k in j['fields'] if k!=j['target_field'])),category='format_failure' if not valid else 'correct' if good else 'source' if source else 'string_null' if v=='null' else 'wrong_null' if v is None else 'control' if v==j.get('control_value') else 'other')
 strict=parse(text);m=re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n\s*```\s*',text,re.S|re.I);semantic=parse(m[1] if m else text)
 return dict(**semantic,strict=strict,outer_fence_removed=bool(m))
def main():
 cfg=read(R/'protocol.json');mc=cfg['models'][KEY];allpairs=read(R/'data/pairs.json')
 assert not (OUT/'pairs.jsonl').exists(),'Refuse overwrite'
 for p,h in read(R/'data/lock.json').items():assert sha(R/p)==h,p
 torch.set_num_threads(8);torch.manual_seed(cfg['seed']);status('loading')
 custom=KEY=='kimi';tok=transformers.AutoTokenizer.from_pretrained(mc['path'],local_files_only=True,trust_remote_code=custom)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 loader=transformers.AutoModelForCausalLM if custom else transformers.AutoModelForImageTextToText
 model=loader.from_pretrained(mc['path'],local_files_only=True,trust_remote_code=custom,torch_dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
 tc=model.config.text_config;count=tc.num_hidden_layers
 candidates=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.ModuleList) and len(m)==count and n.endswith('layers')]
 assert len(candidates)==1,[(n,len(m)) for n,m in candidates]
 layername,layers=candidates[0];assert mc['layers']==[round((count-1)*x/35) for x in [29,33]]
 for k in ['temperature','top_p','top_k']:setattr(model.generation_config,k,None)
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id)
 if mc.get('turn_token'):
  turn=tok.convert_tokens_to_ids(mc['turn_token']);assert turn in tok.all_special_ids;eos.add(turn)
 eos.discard(None);extra={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
 status('hashing_checkpoint');hashes={p.name:sha(p) for p in sorted(Path(mc['path']).iterdir()) if p.suffix in ['.json','.jinja','.safetensors','.py']}
 save('environment.json',dict(model=mc,layer_module=layername,layer_count=count,hidden_size=tc.hidden_size,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),eos=sorted(eos),checkpoint_files=hashes,run_hash=sha(R/'run.py'),chat_template=tok.chat_template))
 def generate(ids,patch=None,fixed=False):
  attempts=[]
  for cap in (mc['replay_budgets'] if fixed else mc['budgets']):
   calls=[];handle=None
   if patch is not None:
    layer,vec=patch
    def hook(module,args,res):
     h=res if torch.is_tensor(res) else res[0]
     if not calls:
      assert h.shape[1]==len(ids);h=h.clone();h[0,-1]=vec.to(h);calls.append(1)
      return h if torch.is_tensor(res) else (h,)+res[1:]
    # 仅在预定位置替换状态；后续沿用接收运行。 / Patch the selected position, then resume the recipient.
    handle=layers[layer].register_forward_hook(hook)
   x=torch.tensor([ids],device='cuda')
   try:
    with torch.inference_mode():out=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,max_new_tokens=cap,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=sorted(eos),**extra)[0,len(ids):].tolist()
   finally:
    if handle:handle.remove()
   if patch is not None:assert len(calls)==1
   attempts.append(dict(cap=cap,token_ids=out,truncated=out[-1] not in eos))
   if out[-1] in eos:break
  return out,attempts
 def decode(ids):
  stop=next((i for i,t in enumerate(ids) if t in eos),len(ids))
  return tok.decode(ids[:stop],skip_special_tokens=False)
 def row(j,kind,ids,attempts,prefix=''):
  generated=prefix+decode(ids);span=answer_span(KEY,generated);raw=generated[slice(*span)] if span else '';s=score(j,raw);tr=attempts[-1]['truncated']
  if tr:
   s.update(correct=False,true_null=False,source_misuse=False,format_valid=False,category='truncated');s['strict'].update(correct=False,true_null=False,source_misuse=False,format_valid=False,category='truncated')
  return dict(uid=j['uid'],kind=kind,raw_output=raw,generated_output=generated,final_answer_found=span is not None,generated_token_ids=ids,truncated=tr,attempts=attempts,**s)
 def context(j,ls):
  prompt=tok.apply_chat_template([dict(role='user',content=j['user_prompt'])],tokenize=False,add_generation_prompt=True,**mc['chat_kwargs']);pids=tok.encode(prompt,add_special_tokens=False)
  ids,attempts=generate(pids);natural=row(j,'natural',ids,attempts);natural['prompt_text']=prompt
  if natural['truncated']:natural['unavailable_reason']='truncated';return natural,None
  prefixes=['']+[decode(ids[:n]) for n in range(1,len(ids)+1)]
  boundary,reason=locate(KEY,natural['generated_output'],j['target_field'],prefixes)
  natural['boundary']=boundary;natural['unavailable_reason']=reason
  if boundary is None:return natural,None
  n,prefix=boundary['n'],boundary['prefix'];inputs=pids+ids[:n];states={};handles=[]
  for l in ls:
   def hook(module,args,res,l=l):states[l]=(res if torch.is_tensor(res) else res[0])[0,-1].detach().cpu().clone()
   handles.append(layers[l].register_forward_hook(hook))
  try:
   with torch.inference_mode():model(input_ids=torch.tensor([inputs],device='cuda'),use_cache=False,**extra)
  finally:
   for h in handles:h.remove()
  out,att=generate(inputs,fixed=True);replay=row(j,'replay',out,att,prefix);replay['natural_equal']=out==ids[n:]
  return natural,dict(j=j,ids=inputs,prefix=prefix,states=states,replay=replay,boundary=boundary)
 def patch(rec,don,l,kind,vec):
  ids,att=generate(rec['ids'],(l,vec),fixed=True);r=row(rec['j'],kind,ids,att,rec['prefix']);r.update(donor_uid=don['j']['uid'],layer=l)
  if kind.startswith('self'):assert ids==rec['replay']['generated_token_ids'],'Self replacement mismatch'
  delta=vec.float()-rec['states'][l].float();real=vec.to(torch.bfloat16).float()-rec['states'][l].float();r.update(intended_delta_norm=float(delta.norm()),realized_delta_norm=float(real.norm()));return r
 dev=[p for p in allpairs if p['B']['dataset']=='round1' and p['B']['split']=='development'];others=[p for p in allpairs if p not in dev]
 if '--smoke' in sys.argv or '--smoke-then-run' in sys.argv:
   smoke=[]
   for j in [dev[0]['B'],dev[0]['D']]:
    n,c=context(j,mc['layers']);smoke.append(dict(natural=n,available=c is not None,self_checks=[patch(c,c,l,'self',c['states'][l]) for l in mc['layers']] if c else []))
   save('smoke.json',smoke);assert any(x['available'] for x in smoke),'No target boundary in smoke';status('smoke_complete')
   if '--smoke-then-run' not in sys.argv:return
 f=(OUT/'pairs.jsonl').open('w',encoding='utf-8',buffering=1);devfile=(OUT/'development_candidates.jsonl').open('w',encoding='utf-8',buffering=1)
 contexts={};devrows=[];statecache={};selected=None;u=None;random_u=None
 def process(p,ls,do_patch=False):
  nb,b=context(p['B'],ls);nd,d=context(p['D'],ls);rows=[nb,nd];boundaries={}
  for c in [b,d]:
   if c:
    rows.append(c['replay']);boundaries[c['j']['condition']]=dict(input_ids=c['ids'],prefix=c['prefix'],**{k:v for k,v in c['boundary'].items() if k!='prefix'});statecache[c['j']['uid']]=c['states']
  if b and d and do_patch:
   l=selected['layer'];bh=b['states'][l].float();dh=d['states'][l].float();delta=bh-dh;proj=u@(u.T@delta);rand=random_u@(random_u.T@delta)
   if float(rand.norm())<1e-12:rand=random_u[:,0]
   rows.extend([patch(b,b,l,'self_B',bh),patch(d,d,l,'self_D',dh),patch(d,b,l,'forward',dh+proj),patch(b,d,l,'reverse',bh-proj),patch(d,b,l,'full',bh),patch(d,b,l,'random',dh+rand*(proj.norm()/rand.norm()))])
  j=p['B'];r=dict(pair_id=p['pair_id'],**{k:j[k] for k in ['dataset','family','template_id','split','form','evidence','target_order','target_field']},gold=j['gold'][j['target_field']],available=bool(b and d),boundaries=boundaries,rows=rows)
  f.write(json.dumps(r,ensure_ascii=False)+'\n');return r,(b,d)
 for i,p in enumerate(dev):
  r,cs=process(p,mc['layers']);devrows.append(r);contexts[p['pair_id']]=cs;status('development',completed=i+1,total=len(dev))
 def rate(form,cond,template=None):
  rr=[p for p in devrows if p['form']==form and p['evidence']=='main' and (template is None or p['template_id']==template)];vals=[next(r for r in p['rows'] if r['kind']=='natural' and r['uid'].endswith('_'+cond)) for p in rr];return sum(r['source_misuse'] for r in vals)/len(vals)
 weak=rate('weak','D')-rate('weak','B');direct=rate('direct','D')-rate('direct','B');ts=sorted({p['template_id'] for p in devrows});diff={t:rate('weak','D',t)-rate('weak','B',t) for t in ts};sr=[r for p in devrows if p['evidence']!='main' for r in p['rows'] if r['kind']=='natural'];acc=sum(r['correct'] for r in sr)/len(sr)
 save('gate.json',dict(passed=weak>=.125 and sum(v>0 for v in diff.values())>=3 and direct<weak and acc>=.9,weak=weak,direct=direct,template_differences=diff,supported_accuracy=acc,diagnostic_only=True))
 ps=[contexts[p['pair_id']] for p in dev if p['B']['form']=='weak'];assert len(ps)==48
 ps=[(b,d) for b,d in ps if b and d]
 counts={e:sum(d['j']['evidence']==e for b,d in ps) for e in ['main','supported','swapped']}
 coverage=dict(available=len(ps),planned=48,evidence=counts,templates=sorted({d['j']['template_id'] for b,d in ps}),coverage_restricted=len(ps)<48)
 save('development_coverage.json',coverage)
 if min(counts.values())>=4 and len(coverage['templates'])==4:
  stats=[];bases={}
  for l in mc['layers']:
   # 不减均值：保留两种指令的整体偏移。 / Uncentered SVD retains the instruction offset.
   mat=torch.stack([b['states'][l].float()-d['states'][l].float() for b,d in ps]);_,sv,vh=torch.linalg.svd(mat,full_matrices=False);torch.save(dict(singular_values=sv,basis=vh[:4].T),OUT/f'svd_{l}.pt')
   for b,d in ps:
    for c in [b,d]:devfile.write(json.dumps(patch(c,c,l,'self',c['states'][l]),ensure_ascii=False)+'\n')
   for rank in [1,4]:
    basis=vh[:rank].T.contiguous();bases[l,rank]=basis;rs=[]
    for b,d in ps:
     dh=d['states'][l].float();delta=b['states'][l].float()-dh;r=patch(d,b,l,'development_forward',dh+basis@(basis.T@delta));r['rank']=rank;rs.append(r);devfile.write(json.dumps(r,ensure_ascii=False)+'\n')
    gains={e:sum(int(r['correct'])-int(d['replay']['correct']) for r,(b,d) in zip(rs,ps) if d['j']['evidence']==e) for e in ['main','supported','swapped']}
    nullgain=sum(int(r['true_null'])-int(d['replay']['true_null']) for r,(b,d) in zip(rs,ps) if d['j']['evidence']=='main');reduction=sum(int(d['replay']['source_misuse'])-int(r['source_misuse']) for r,(b,d) in zip(rs,ps) if d['j']['evidence']=='main')
    stats.append(dict(layer=l,rank=rank,gains=gains,null_gain=nullgain,source_reduction=reduction,eligible=reduction>0 and gains['supported']>=-1 and gains['swapped']>=-1,macro_gain=sum(gains[e]/counts[e] for e in counts)/3))
  elig=[s for s in stats if s['eligible']];selected=sorted(elig,key=lambda s:(-s['null_gain'],s['rank'],s['layer']))[0] if elig else sorted(stats,key=lambda s:(-s['macro_gain'],s['rank'],s['layer']))[0]
  u=bases[selected['layer'],selected['rank']];random_u=torch.linalg.qr(torch.randn(u.shape,generator=torch.Generator().manual_seed(cfg['seed']))).Q;torch.save(u,OUT/'frozen_basis.pt');torch.save(random_u,OUT/'random_basis.pt')
  save('selection.json',dict(selected=selected,candidates=stats,validation_seen=False,coverage=coverage,basis_sha256=sha(OUT/'frozen_basis.pt')))
 else:save('selection.json',dict(selected=None,validation_seen=False,reason='Insufficient locatable development coverage',coverage=coverage))
 torch.save(statecache,OUT/'development_states.pt');statecache={};contexts.clear();status('selection_frozen',selected=selected)
 for i,p in enumerate(others):
  process(p,[selected['layer']] if selected else mc['layers'],selected is not None)
  if (i+1)%40==0:torch.save(statecache,OUT/'evaluation_states.pt')
  status('evaluation',completed=i+1,total=len(others))
 torch.save(statecache,OUT/'evaluation_states.pt');f.close();devfile.close();status('complete',pairs=len(allpairs),intervention_run=selected is not None)
if __name__=='__main__':
 try:main()
 except Exception:status('failed',error=traceback.format_exc());raise

"""Limited lexical readout and value diagnostics; no causal intervention."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parent
OLD=Path('./jspace_task_selection')
sys.path.insert(0,str(OLD/'official/jacobian-lens-main'))
import jlens
import torch
import transformers
from run import status,write

def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 assert read('results/gate_decision.json')['pass']
 cfg=read('protocol.json');audit=read('results/lens_token_audit.json')
 torch.manual_seed(cfg['seed']);torch.set_num_threads(8)
 tok=transformers.AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True);tok.padding_side='left';tok.pad_token=tok.eos_token
 ids={k:v[0] for k,v in audit['tokens'].items() if len(v)==1}
 assert len(ids)==7
 protocol=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='Locked after behavior, before any internal readout',
  layers=[9,13,17,21,25,29,33],final_layer=35,selection='Two largest absolute development PN paired D-B office-minus-null lexical logit differences. No validation-based layer selection.',
  positions=dict(P0='Last native-chat prompt token before JSON generation.',PN='Replay exact generated token prefix, last boundary after target-key colon and before value content. No candidate value token included.',
   P1='Canonical forced JSON prefix through target-key colon. Prior auxiliary field is set to gold; diagnostic, not natural output.'),
  scope='P0/PN: all weak B/D groups, both orders. P1: first entity of every template, weak/direct/missing/supported B/D both orders.',
  tokens=ids,readouts=['official J-lens full unembedding','same-position Logit Lens'],
  metrics=['office-minus-null lexical logits','office-minus-launch lexical logits','full-vocabulary and six-city ranks'],
  value_scoring='P1 only: final-model conditional log probability of complete quoted office city vs JSON null, with identical next delimiter. This is a forced-prefix value preference, not a natural generation rate.',
  limitations=['City lexical readout does not encode proof of the headquarters relation.','Natural-prefix replay is observational and may differ from original batched generation at rounding level.',
   'Layer selection and internal analysis planned after behavioral outcomes were observed; internal held-out templates are not new behavior samples.',
   'Lens calibrated on 100 independent English passages of at most 128 tokens; no task-specific calibration-quality guarantee.'],
  provenance=audit['lens_metadata'],model_revision=read('results/environment_main.json')['model_revision'])
 write('results/lens_protocol.json',protocol)
 status('lens_loading')
 hf=transformers.AutoModelForCausalLM.from_pretrained(cfg['model_path'],torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).cuda().eval()
 hf.requires_grad_(False);model=jlens.from_hf(hf,tok,force_bos=False)
 lens=jlens.JacobianLens.load(str(OLD/'results/jacobian_lens.pt'))
 lens.jacobians={l:j.cuda() for l,j in lens.jacobians.items() if l in protocol['layers']}
 records=[json.loads(x) for s in ['development','validation'] for x in (ROOT/f'results/{s}.jsonl').read_text().splitlines()]
 contexts=[];coverage=[]
 for r in records:
  if r['condition'] not in ['B','D']:continue
  prompt_ids=tok.encode(r['prompt_text'],add_special_tokens=False)
  assert len(prompt_ids)==r['input_tokens']
  base={k:r[k] for k in ['job_id','group_id','template_id','split','relation_form','condition','target_order','office_city','launch_city','target_category','office_city_misuse']}
  if r['relation_form']=='weak':
   contexts.append(dict(**base,position='P0',input_ids=prompt_ids,prefix='',token_index=len(prompt_ids)-1))
   match=re.search(r'"总部城市"\s*:',r['raw_output'])
   colon=match.end() if match else -1
   limit=colon+len(r['raw_output'][colon:])-len(r['raw_output'][colon:].lstrip()) if match else -1
   boundaries=[]
   if match:
    for n in range(1,len(r['generated_token_ids'])+1):
     dec=tok.decode(r['generated_token_ids'][:n],skip_special_tokens=False)
     if colon<=len(dec)<=limit:boundaries.append((n,dec))
     if len(dec)>limit:break
   coverage.append(dict(job_id=r['job_id'],available=bool(boundaries),reason='exact_prevalue_boundary' if boundaries else 'no_boundary_without_value_content'))
   if boundaries:
    n,prefix=boundaries[-1];tokens=prompt_ids+r['generated_token_ids'][:n]
    contexts.append(dict(**base,position='PN',input_ids=tokens,prefix=prefix,token_index=len(tokens)-1))
  if r['group_id'].endswith('e1') and r['relation_form'] in ['weak','direct','missing','supported']:
   prior={}
   for field in r['fields']:
    if field==r['target_field']:break
    prior[field]=r['gold'][field]
   prefix='{' if not prior else json.dumps(prior,ensure_ascii=False)[:-1]+', '
   prefix+='"总部城市":'
   text=r['prompt_text']+prefix;tokens=tok.encode(text,add_special_tokens=False)
   contexts.append(dict(**base,position='P1',input_ids=tokens,prefix=prefix,token_index=len(tokens)-1,text=text))
 write('results/lens_context_audit.json',dict(n_contexts=len(contexts),natural_boundary_coverage=coverage,
  contexts=[{k:v for k,v in c.items() if k not in ['input_ids','text']} for c in contexts]))
 out=[]
 def paired(rows,position,layer,method):
  rs=[r for r in rows if r['position']==position and r['layer']==layer and r['method']==method and r['relation_form']=='weak']
  lookup={(r['group_id'],r['condition'],r['target_order']):r['office_minus_null'] for r in rs}
  values=[]
  for g in sorted({r['group_id'] for r in rs}):
   for order in ['first','last']:
    if (g,'D',order) in lookup and (g,'B',order) in lookup:values.append(lookup[g,'D',order]-lookup[g,'B',order])
  return dict(mean_D_minus_B=sum(values)/len(values) if values else None,n_pairs=len(values))
 def forward(selected,layers,split):
  for start in range(0,len(selected),8):
   chunk=selected[start:start+8];inputs=tok.pad([dict(input_ids=c['input_ids'],attention_mask=[1]*len(c['input_ids'])) for c in chunk],padding=True,return_tensors='pt').to('cuda')
   residuals={};handles=[]
   for l in layers:
    def hook(module,ins,result,l=l):residuals[l]=(result if torch.is_tensor(result) else result[0])[:,-1].detach()
    handles.append(model.layers[l].register_forward_hook(hook))
   try:
    with torch.inference_mode():hf(**inputs,use_cache=False,logits_to_keep=1)
   finally:
    for h in handles:h.remove()
   with torch.inference_mode():
    for l in layers:
     methods=['Logit Lens'] if l==35 else ['J-lens','Logit Lens']
     for method in methods:
      h=residuals[l];h=lens.transport(h.float(),l) if method=='J-lens' else h
      logits=model.unembed(h).float()
      city_ids=[ids[c] for c in ids if c!='null']
      for i,c in enumerate(chunk):
       office=logits[i,ids[c['office_city']]].item();launch=logits[i,ids[c['launch_city']]].item();null=logits[i,ids['null']].item()
       row={k:v for k,v in c.items() if k not in ['input_ids','text','prefix']}
       row.update(layer=l,method=method,office_logit=office,launch_logit=launch,null_logit=null,office_minus_null=office-null,office_minus_launch=office-launch,
        office_vocab_rank=1+int((logits[i]>office).sum()),office_city_rank=1+int((logits[i,city_ids]>office).sum()))
       out.append(row)
   if start%40==0:status('lens_'+split,completed=min(start+8,len(selected)),total=len(selected))
 dev=[c for c in contexts if c['split']=='development']
 forward(dev,protocol['layers']+[35],'development')
 differences={l:paired(out,'PN',l,'J-lens') for l in protocol['layers']}
 chosen=sorted(protocol['layers'],key=lambda l:(-abs(differences[l]['mean_D_minus_B'] or 0),l))[:2]
 write('results/lens_layer_selection.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selected_layers=chosen,development_differences=differences,validation_readouts_seen=False))
 val=[c for c in contexts if c['split']=='validation']
 forward(val,chosen+[35],'validation')
 with (ROOT/'results/lens_readouts.jsonl').open('w',encoding='utf-8') as f:
  for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 # Score exact complete candidate values in controlled P1 contexts only.
 scoring=[]
 for c in contexts:
  if c['position']!='P1':continue
  delimiter=',' if c['target_order']=='first' else '}'
  for kind,value in [('office',c['office_city']),('null',None)]:
   continuation=' '+json.dumps(value,ensure_ascii=False)+delimiter
   full=tok.encode(c['text']+continuation,add_special_tokens=False)
   assert full[:len(c['input_ids'])]==c['input_ids'],'Unstable prefix tokenization'
   scoring.append(dict(context=c,kind=kind,tokens=full,n_value=len(full)-len(c['input_ids']),continuation=continuation))
 value_rows=[]
 for start in range(0,len(scoring),12):
  chunk=scoring[start:start+12];inputs=tok.pad([dict(input_ids=x['tokens'],attention_mask=[1]*len(x['tokens'])) for x in chunk],padding=True,return_tensors='pt').to('cuda')
  keep=max(x['n_value'] for x in chunk)+1
  with torch.inference_mode():
   logits=hf(**inputs,use_cache=False,logits_to_keep=keep).logits.float().log_softmax(-1)
  for i,x in enumerate(chunk):
   n=x['n_value'];target=torch.tensor(x['tokens'][-n:],device='cuda')
   logp=logits[i,-n-1:-1].gather(-1,target[:,None]).sum().item()
   c=x['context'];value_rows.append({**{k:v for k,v in c.items() if k not in ['input_ids','text','prefix']},'candidate':x['kind'],'continuation':x['continuation'],'logp':logp})
 with (ROOT/'results/controlled_value_scores.jsonl').open('w',encoding='utf-8') as f:
  for r in value_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 summary=[]
 for split in ['development','validation']:
  subset=[r for r in out if r['split']==split]
  for pos in ['P0','PN','P1']:
   for l in (protocol['layers'] if split=='development' else chosen):
    for method in ['J-lens','Logit Lens']:summary.append(dict(split=split,position=pos,layer=l,method=method,**paired(subset,pos,l,method)))
 write('results/lens_summary.json',dict(selected_layers=chosen,paired_lexical_differences=summary,natural_coverage=sum(x['available'] for x in coverage),natural_requested=len(coverage),n_contexts=len(contexts),n_readouts=len(out),n_controlled_value_scores=len(value_rows),interpretation='Descriptive lexical representations and forced-prefix value preference only. No causal intervention or relation-specific direction identified.'))
 status('lens_complete',selected_layers=chosen,n_contexts=len(contexts),n_readouts=len(out))
if __name__=='__main__':
 try:main()
 except Exception as exc:status('failed',error=repr(exc),traceback=traceback.format_exc());raise

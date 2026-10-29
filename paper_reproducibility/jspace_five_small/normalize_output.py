"""Separate native assistant turn terminator, without editing answer content."""
import json,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from scoring import score
from analyze import analyze
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'protocol.json').read_text());tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True)
turn=tok.convert_tokens_to_ids('<|im_end|>');assert turn in tok.all_special_ids
assert tok.decode([turn],skip_special_tokens=False)=='<|im_end|>'
rawpath=ROOT/'results/outputs.jsonl';raw=[json.loads(x) for x in rawpath.read_text().splitlines()];out=[];tails={}
for r in raw:
 ids=r['generated_token_ids'];assert turn in ids,(r['job_id'],'missing native turn terminator')
 n=ids.index(turn);body=tok.decode(ids[:n],skip_special_tokens=False)
 tail=tok.decode(ids[n:],skip_special_tokens=False);assert not tok.decode(ids[n:],skip_special_tokens=True).strip(),(r['job_id'],tail)
 assert r['raw_output'].startswith(body+'<|im_end|>')
 assert r['raw_output'][len(body):].replace('<|im_end|>','').strip()==''
 updated={**r,**score(r,body,False),'raw_output':body,'raw_output_with_transport_suffix':r['raw_output'],'native_turn_end_id':turn,'answer_token_count':n}
 out.append(updated);tails[tail]=tails.get(tail,0)+1
with (ROOT/'results/answer_outputs.jsonl').open('w',encoding='utf-8') as f:
 for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
audit=dict(raw_preserved=True,n=len(raw),native_turn_end_id=turn,special_token_verified=True,removed_only_native_turn_end_and_terminal_whitespace_and_eos=True,tail_counts=tails,raw_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),normalized_sha256=hashlib.sha256((ROOT/'results/answer_outputs.jsonl').read_bytes()).hexdigest(),interpretation='Transport framing normalization only, no answer repair. Generation was not rerun. Original scoring erroneously included native turn terminator in JSON body.')
(ROOT/'results/transport_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'results/answer_summary.json').write_text(json.dumps(analyze(out,cfg),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))

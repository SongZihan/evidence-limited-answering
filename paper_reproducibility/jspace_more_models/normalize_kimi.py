import json,importlib,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
audit=json.loads((ROOT/'results/kimi_transport_audit.json').read_text())
assert audit['n']==3516 and audit['all_bodies_equal_original_minus_single_terminal_marker']
for ds in ['company','four','five']:
 p=ROOT/f'results/kimi16b_{ds}.jsonl';score=importlib.import_module(ds+'_scoring').score;analyze=importlib.import_module(ds+'_analyze').analyze
 rows=[]
 for line in p.read_text(encoding='utf-8').splitlines():
  r=json.loads(line);assert r['raw_output'].endswith('<|im_end|>');assert r['generated_token_ids'][-2:]==[163586,163585]
  body=r['raw_output'][:-len('<|im_end|>')]
  rows.append({**r,**score(r,body,r['truncated_generation']),'transport_raw_output':r['raw_output'],'raw_output':body,'native_turn_separated':True})
 (ROOT/f'results/kimi16b_{ds}_native.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
 cfg=json.loads((ROOT/f'data/{ds}_config.json').read_text(encoding='utf-8'))
 summary={s:analyze([r for r in rows if r['split']==s],cfg) for s in ['development','validation']} if ds=='company' else analyze(rows,cfg)
 (ROOT/f'results/kimi16b_{ds}_native_strict_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(ds,len(rows),'invalid',sum(not r['format_valid'] for r in rows))

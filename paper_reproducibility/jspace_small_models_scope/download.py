import json,time,sys,traceback,inspect
from pathlib import Path
from modelscope import snapshot_download
R=Path(__file__).resolve().parent;KEY=sys.argv[1];cfg=json.loads((R/'protocol.json').read_text())['models'][KEY]
def save(stage,**kw):(R/f'results/{KEY}_download.json').write_text(json.dumps(dict(stage=stage,repo=cfg['repo'],utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw),ensure_ascii=False,indent=2),encoding='utf-8')
try:
 save('probing');print(inspect.signature(snapshot_download),flush=True)
 # Read only metadata and model card first; do not download alternate quantizations/formats.
 p=snapshot_download(cfg['repo'],local_dir=cfg['path'],allow_patterns=['config.json','generation_config.json','README.md','tokenizer_config.json','special_tokens_map.json'])
 c=json.loads((Path(p)/'config.json').read_text());save('metadata_ready',config=c)
 if '--probe' in sys.argv:sys.exit(0)
 save('downloading',config=c)
 p=snapshot_download(cfg['repo'],local_dir=cfg['path'],allow_patterns=['*.json','*.safetensors','*.model','*.tiktoken','*.jinja','*.txt','README.md'])
 assert list(Path(p).glob('*.safetensors')),'No original safetensors downloaded'
 save('complete',path=p,config=c)
except Exception:save('failed',error=traceback.format_exc());raise

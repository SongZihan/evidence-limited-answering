"""Token and provenance audit before choosing or inspecting any lens readout."""
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
MODEL=Path('models/Qwen3-8B')
OLD=Path('./jspace_task_selection/results')
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
meta=json.loads((OLD/'lens_metadata.json').read_text())
assert hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest()==meta['model_config_sha256']
assert hashlib.sha256((OLD/'jacobian_lens.pt').read_bytes()).hexdigest()==meta['lens_sha256']
cities=['武汉','上海','南京','成都','杭州','北京']
audit=dict(tokens={c:tok.encode(c,add_special_tokens=False) for c in cities+['null']},
 lens_metadata=meta,model_config_matches=True,lens_hash_matches=True,
 scope='Bare city tokens are lexical probes. Multitoken cities are excluded from full-city lens claims before reading scores; behavioral rows retained.')
(ROOT/'results/lens_token_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))

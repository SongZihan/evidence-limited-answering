"""Link user-supplied checkpoints on Linux; never overwrite existing paths."""
from pathlib import Path
import argparse
import json

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
args = parser.parse_args()
mapping = json.loads(Path(args.config).read_text('utf-8'))
for name, source in mapping.items():
    if not name.startswith('models/') or '..' in Path(name).parts:
        raise ValueError('Checkpoint keys must stay inside models/')
    target = root / name
    source = Path(source).expanduser().resolve()
    if not source.is_dir() or source == target.resolve():
        raise ValueError(f'Provide an existing external checkpoint directory for {name}')
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(name)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
    print(f'Linked {name}')

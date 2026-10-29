"""Restore the verified reproduction package using Python's standard library."""
from pathlib import Path
import hashlib
import json
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / 'data_manifest.json').read_text('utf-8'))
    # 先校验全部分块，再还原，避免将不完整数据用于复现。
    # Verify every part before extraction; never overwrite a changed local file.
    with tempfile.TemporaryDirectory(prefix='evidence-limited-') as tmp:
        archive = Path(tmp) / manifest['archive_name']
        with archive.open('wb') as output:
            for part in manifest['parts']:
                path = ROOT / part['path']
                if not path.is_file():
                    raise FileNotFoundError(f'Missing {part["path"]}; download or clone the complete repository.')
                assert path.stat().st_size == part['size'], part['path']
                assert sha(path) == part['sha256'], part['path']
                output.write(path.read_bytes())
        assert sha(archive) == manifest['archive_sha256'], 'Archive checksum mismatch'
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None, 'Archive integrity failure'
            # Preflight all names/conflicts before writing any file.
            for entry in z.infolist():
                target = (ROOT / entry.filename).resolve()
                assert target.is_relative_to(ROOT / 'paper_reproducibility'), entry.filename
                if target.is_file():
                    local, archived = target.read_bytes(), z.read(entry)
                    text_file = target.suffix in {'.py', '.md', '.txt', '.json', '.sha256'} or target.name.endswith('LICENSE')
                    same_text = text_file and local.replace(b'\r\n', b'\n') == archived.replace(b'\r\n', b'\n')
                    if local != archived and not same_text:
                        raise FileExistsError(f'Local modifications found: {entry.filename}; use a fresh checkout.')
            z.extractall(ROOT)
    print('Restored and verified paper_reproducibility/.')
    print('Next: cd paper_reproducibility && python reproduce.py verify')


if __name__ == '__main__':
    main()

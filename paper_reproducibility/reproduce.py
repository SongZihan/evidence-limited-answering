"""CPU reproduction entry point. Run from any working directory."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'formal/_draft_work'
OUTPUT = ROOT / 'reproduced'


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                   cwd=ROOT, check=True)


def scientific(value):
    """文件路径及校验值会因脱敏变化；实验数值必须保持一致。
    Compare scientific content without machine-specific provenance hashes.
    """
    ignored = {'sha256', 'source_hashes', 'sources', 'status', 'source',
               'environment', 'run_sha256', 'code_sha256'}
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items()
                if k not in ignored and not k.endswith('_sha256')}
    if isinstance(value, list):
        return [scientific(v) for v in value]
    return value


def summarize():
    for section in [41, 42, 43, 44]:
        run(f'formal/_draft_work/section{section}_data.py')
        name = f'section{section}_data.json'
        actual = json.loads((WORK / name).read_text('utf-8'))
        expected = json.loads((ROOT / 'expected' / name).read_text('utf-8'))
        assert scientific(actual) == scientific(expected), name
    run('jspace_core_expanded/report.py')
    actual = json.loads((ROOT / 'jspace_core_expanded/results/summary.json').read_text('utf-8'))
    expected = json.loads((ROOT / 'expected/expanded_summary.json').read_text('utf-8'))
    assert scientific(actual) == scientific(expected), 'expanded validation'
    print('PASS: row-level summaries match the saved publication numbers.')


def plots():
    import cairosvg
    OUTPUT.mkdir(exist_ok=True)
    base = OUTPUT / 'readout_input'
    base.mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'jspace_relation_mechanism_round1/data/jobs.json', base / 'jobs.json')
    shutil.copy2(ROOT / 'jspace_relation_mechanism_round1/results/readouts.jsonl', base / 'readouts.jsonl')
    run('formal/_draft_work/plot_section32.py', ROOT / 'jspace_relation_binding/results/validation.jsonl', OUTPUT)
    run('formal/_draft_work/plot_section34.py', base, OUTPUT)
    for section in [41, 42, 43]:
        run(f'formal/_draft_work/plot_section{section}.py', WORK / f'section{section}_data.json', OUTPUT)
    run('formal/_draft_work/plot_fig8_expanded.py', WORK / 'section44_data.json',
        ROOT / 'jspace_core_expanded/results/summary.json', OUTPUT)
    names = ['instruction_example', 'behavior_effect', 'internal_readouts', 'model_comparison',
             'scenario_effects', 'validation_readouts', 'intervention_selectivity']
    # Plot scripts predate manuscript renumbering; map them explicitly.
    for number, name in enumerate(names, 1):
        target = OUTPUT / f'figure{number}_{name}_en.svg'
        if number == 1:
            shutil.copy2(ROOT / 'en_img' / target.name, target)
        else:
            source = OUTPUT / f'figure{number+1}_{name}_en.svg'
            assert source.exists(), source
            shutil.copy2(source, target)
    run('tools/layout_figures.py', OUTPUT)
    for number, name in enumerate(names, 1):
        source = OUTPUT / f'figure{number}_{name}_en.svg'
        cairosvg.svg2png(url=str(source), write_to=str(source.with_suffix('.png')), output_width=3200)
    print('PASS: seven English SVG/PNG pairs regenerated.')


def verify():
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text('utf-8-sig'), filename=str(p.relative_to(ROOT)))
    for p in ROOT.rglob('*.json'):
        json.loads(p.read_text('utf-8-sig'))
    for p in ROOT.rglob('*.jsonl'):
        for line in p.read_text('utf-8-sig').splitlines():
            if line.strip():
                json.loads(line)
    manifest = ROOT / 'MANIFEST.sha256'
    if manifest.exists():
        for line in manifest.read_text('utf-8').splitlines():
            digest, name = line.split('  ', 1)
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    print('PASS: syntax, JSON records and distribution checksums.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'summarize', 'plots', 'all'])
    args = parser.parse_args()
    if args.command in ['verify', 'all']:
        verify()
    if args.command in ['summarize', 'all']:
        summarize()
    if args.command in ['plots', 'all']:
        plots()

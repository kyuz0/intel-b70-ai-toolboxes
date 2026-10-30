#!/usr/bin/env python3
"""Prepare AtomicChat Flash Next and the official MTP tensors for Strata."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if not a.model.is_file():ap.error('Download the full AtomicChat shard set first')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    tools=Path('/opt/strata/tools')
    def run(name,*args):
        print(f'Preparing {name}',flush=True)
        subprocess.run([sys.executable,str(tools/name),*map(str,args)],check=True)
    run('iq_pack.py','--gguf',a.model,'--out',out/'pack','--compat-bf16')
    run('mtp_fetch.py','fetch','--out',out/'mtp')
    run('mtp_pack.py','--src',out/'mtp','--experts','q2_0','--out',out/'mtp/mtp-q2_0.gguf')
    run('mtp_rt.py','--gguf',out/'mtp/mtp-q2_0.gguf','--out',out/'mtp/rt')
    files={str(p.relative_to(out)):p.stat().st_size for sub in ('pack','mtp/rt') for p in (out/sub).rglob('*') if p.is_file()}
    receipt={'schema_version':1,'strata_source':Path('/opt/b70-provenance/strata-source.txt').read_text().strip(),
        'main_revision':'142262902a46f7daed19c79d0771534c8106ad59',
        'mtp_revision':'de4b8e4d43b917e7706784d8bb445c9af86a3540',
        'conversion':'compat-bf16; MTP q2_0','files':files}
    (out/'preparation.json.tmp').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'preparation.json.tmp').replace(out/'preparation.json')
    print('Strata model preparation complete.',flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Start Strata's Intel HTTP server with the qualified dual-B70 Flash Next profile."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='First AtomicChat AD4.27 GGUF shard')
    parser.add_argument('--pack', type=Path, required=True, help='Prepared native pack directory')
    parser.add_argument('--mtp', type=Path, help='Prepared official Q2_0 MTP runtime directory; omit for baseline')
    parser.add_argument('--context', type=int, default=68096)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 2048 <= args.context <= 68096:
        parser.error('Context must be between 2048 and 68096 for this qualified profile')
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    if not args.model.is_file() or not (args.pack / 'index.txt').is_file():
        parser.error('Download the complete model and prepare its native pack first')
    if not (args.pack / 'tokenizer' / 'chat_template.jinja').is_file():
        parser.error('Prepared pack is missing its chat template')
    if args.mtp is not None and not args.mtp.is_dir():
        parser.error('Prepare the official MTP runtime directory first')
    engine_args = ['--pack', str(args.pack.resolve()), '--native', str(args.model.resolve()),
        '--layer-split', 'auto', '--expert-cache', 'auto', '--resident-experts',
        '--expert-profile', '/opt/strata/data/expert-profile-sycl.bin',
        '--prefill', 'auto:4096', '--max-context', str(args.context), '--kv', 'int8',
        '--kv-resident', str(min(args.context, 32768)), '--vram-reserve-mib', '1536',
        '--suffix-draft', '0', '--spec', '4' if args.mtp else '2']
    if args.mtp:
        engine_args += ['--mtp', str(args.mtp.resolve()), '--spec-min-p', '0.5', '--coupled-draft']
    config = {'exe':'/opt/strata/build/strata', 'cwd':'/opt/strata', 'args':engine_args,
        'tokenizer':str((args.pack/'tokenizer').resolve()), 'model_name':'Qwen3.8-Flash-Next',
        'arch':'qwen4exp', 'backend':'sycl', 'parallel':1, 'open_browser':False,
        'log':'/tmp/strata-engine.log',
        'env':{'ONEAPI_DEVICE_SELECTOR':'level_zero:*', 'SYCL_CACHE_PERSISTENT':'0', 'STRATA_MIRROR_MIB':'0'}}
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',prefix='strata-serve-',delete=False) as file:
        json.dump(config,file)
        config_path=file.name
    os.execv(sys.executable,[sys.executable,'/opt/strata/sycl/serve/server_intel.py',
        '--engine','strata','--config',config_path,'--host',args.host,'--port',str(args.port),
        '--no-api-key','--offline'])

if __name__ == '__main__':
    main()

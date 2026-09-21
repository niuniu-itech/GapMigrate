#!/usr/bin/env python3
"""Optional checked packing-loop transformation, separate from six BLAS variants."""
import argparse
import hashlib
import json
from pathlib import Path
from gapmigration.scoped_packing import generate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('experiments/openblas/auxiliary/packing.cpp'))
    p.add_argument('--profiles',type=Path,default=Path('experiments/openblas/configs/profiles.json'))
    p.add_argument('--profile',choices=['P0','P1','P2','P3','P4'],default='P3')
    p.add_argument('--lmul',type=int,choices=[1,2,4,8],default=1)
    p.add_argument('--mode',choices=['gather','tile'],default='tile')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();profile=next(x for x in json.loads(args.profiles.read_text()) if x['id']==args.profile)
    if args.output.exists() and any(args.output.iterdir()):p.error('Output must be empty')
    code,certificate=generate(args.source,profile,args.lmul,args.mode)
    text=args.source.read_text();notice=text.split('*/',1)[0]+'*/\n' if text.lstrip().startswith('/*') else ''
    code=notice+code;args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'packing.cpp').write_text(code,encoding='utf-8',newline='\n')
    certificate.update(published_sha256=hashlib.sha256(code.encode()).hexdigest(),status='constructed_not_qualified')
    (args.output/'certificate.json').write_text(json.dumps(certificate,indent=2))

if __name__=='__main__':main()

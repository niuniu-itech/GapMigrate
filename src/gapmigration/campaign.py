"""Construct the registered candidate pool from preprocessed source kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .frontend import Unsupported
from .mapping import level2,level3,materialize
from .mapping.common import load_source,source_digest

VARIANTS=('gemm','trmm','gemv_n','gemv_t','symv_L','symv_U')

def generate(kernel_dir,profiles,output,variants=VARIANTS):
    kernel_dir=Path(kernel_dir);output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Candidate output must be empty')
    (output/'sources').mkdir(parents=True,exist_ok=True)
    (output/'analysis').mkdir()
    catalog=[];rejections=[];started=time.monotonic()
    for variant in variants:
        if variant not in VARIANTS:raise ValueError('unsupported variant '+variant)
        path=kernel_dir/(variant+'.cpp');ast,notice=load_source(path)
        for profile in profiles:
            builder=level3 if variant in ('gemm','trmm') else level2
            rows,rejected,scope=builder.construct(ast,variant,profile)
            if builder is level2:
                for row in list(rows):
                    if row['organization']=='retain' and profile['id'] in ('P3','P4'):
                        try:rows.append(materialize.construct(row))
                        except Unsupported as exc:rejected.append(dict(id=row['id']+'_materialize',reason=str(exc)))
            analysis_path='analysis/'+variant+'_'+profile['id']+'.json'
            (output/analysis_path).write_text(json.dumps(scope,indent=2),encoding='utf-8')
            for row in rows:
                code=notice+row.pop('code');target='sources/'+row['id']+'.cpp'
                (output/target).write_text(code,encoding='utf-8',newline='\n')
                row.update(path=target,source_sha256=source_digest(path),sha256=hashlib.sha256(code.encode()).hexdigest(),
                    analysis=analysis_path,status='constructed_not_qualified',contract='registered FP32, VLEN=256, nonaliasing positive-size inputs')
                catalog.append(row)
            rejections.extend(rejected)
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
    (output/'rejections.json').write_text(json.dumps(rejections,indent=2),encoding='utf-8')
    (output/'construction.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-started,
        candidates=len(catalog),rejections=len(rejections),scope='source analysis and candidate construction, separate from search budget'),indent=2))
    return catalog,rejections

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernels',type=Path,default=Path('experiments/openblas/kernels'))
    p.add_argument('--profiles',type=Path,default=Path('experiments/openblas/configs/profiles.json'))
    p.add_argument('--variant',choices=VARIANTS,action='append')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    rows,rejected=generate(args.kernels,json.loads(args.profiles.read_text()),args.output,args.variant or VARIANTS)
    print(json.dumps(dict(candidates=len(rows),rejected=len(rejected),output=str(args.output))))
    return 0

if __name__=='__main__':
    raise SystemExit(main())

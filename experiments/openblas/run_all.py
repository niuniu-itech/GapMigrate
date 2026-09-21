#!/usr/bin/env python3
"""One-command build and verification of six kernels and six migrated examples."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
EXAMPLES={
    'gemm':'P1_gemm_L1_group4',
    'trmm':'trmm_RU_P1_m1_g4',
    'gemv_n':'gemv_n_P1_m1_retain',
    'gemv_t':'gemv_t_P4_m1_chunk_reduce',
    'symv_L':'symv_L_P4_m1_materialize',
    'symv_U':'symv_U_P4_m1_materialize',
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/openblas_suite_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--cxx',default='g++')
    p.add_argument('--suite',choices=['both','source','migrated'],default='both')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--build-only',action='store_true')
    mode.add_argument('--run-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cpu',type=int)
    args=p.parse_args()
    if args.repeats<1:p.error('--repeats must be positive')
    if not args.build_only and (platform.system()!='Linux' or platform.machine().lower() not in ('riscv64','riscv')):
        p.error('Execution requires physical RVV Linux. On x86-64 use --build-only with a RISC-V cross compiler.')
    out=Path(args.output).resolve()
    if args.run_only and not out.is_dir():p.error('--run-only requires the transferred build directory')
    out.mkdir(parents=True,exist_ok=True)
    report=out/('run_summary.json' if args.run_only else 'build_summary.json' if args.build_only else 'summary.json')
    if report.exists():p.error('Summary already exists. Preserve it and choose a new output directory.')
    subprocess.run([sys.executable,str(ROOT/'run.py'),'verify'],check=True)
    rows=[];failed=False
    for variant,candidate in EXAMPLES.items():
        for kind in ('source','migrated'):
            if args.suite not in ('both',kind):continue
            directory=out/(variant+'_'+kind)
            row=dict(variant=variant,kind=kind,candidate=candidate if kind=='migrated' else None)
            if not args.run_only:
                cmd=[sys.executable,str(ROOT/'run.py'),'build','--output',str(directory),'--cxx',args.cxx]
                cmd+=['--variant',variant] if kind=='source' else ['--candidate',candidate]
                r=subprocess.run(cmd,text=True,capture_output=True)
                row['build']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
            if not args.build_only and (args.run_only or row['build']['returncode']==0):
                cmd=[sys.executable,str(ROOT/'run.py'),'run','--build',str(directory),'--repeats',str(args.repeats)]
                if args.smoke:cmd+=['--smoke']
                if args.cpu is not None:cmd+=['--cpu',str(args.cpu)]
                r=subprocess.run(cmd,text=True,capture_output=True)
                row['run']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
            row['passed']=all(v['returncode']==0 for k,v in row.items() if k in ('build','run'))
            failed|=not row['passed'];rows.append(row)
            report.write_text(json.dumps(dict(scope='six preprocessed kernels and six representative pool candidates; not a search replay',
                mode='build_only' if args.build_only else 'run_only' if args.run_only else 'build_and_run',
                smoke=args.smoke,repeats=args.repeats,jobs=rows),indent=2))
            print(variant,kind,'PASS' if row['passed'] else 'FAIL',flush=True)
    print('Summary:',report)
    return int(failed)

if __name__=='__main__':
    sys.exit(main())

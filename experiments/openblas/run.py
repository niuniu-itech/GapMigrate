#!/usr/bin/env python3
"""Build isolated RVV kernels, then validate/time on physical RVV Linux."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FLAGS = ['-O3', '-std=c++11', '-march=rv64gcv_zvl256b', '-mabi=lp64d',
         '-fno-fast-math', '-ffp-contract=off', '-fno-tree-vectorize',
         '-fno-tree-slp-vectorize', '-fno-tree-loop-distribute-patterns',
         '-fno-builtin', '-fno-stack-protector']
VARIANTS = ['gemm','trmm','gemv_n','gemv_t','symv_L','symv_U']

def read(relative):
    return json.loads((ROOT/relative).read_text())

def run_command(cmd, timeout):
    try:
        r = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, timeout=timeout)
        return dict(command=[str(x) for x in cmd], returncode=r.returncode,
                    stdout=r.stdout, stderr=r.stderr)
    except subprocess.TimeoutExpired:
        return dict(command=[str(x) for x in cmd], returncode=None, status='timeout')

def verify():
    records=read('kernels/manifest.json')+read('candidates/catalog.json')
    for r in records:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    cases=read('configs/inputs.json')
    for v in VARIANTS:
        assert sum(c['variant']==v for c in cases)==15,v
    print(json.dumps(dict(status='verified',sources=6,candidates=len(records)-6,inputs=len(cases))))

def build(args):
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory to preserve previous builds and results.')
    out.mkdir(parents=True,exist_ok=True)
    variant=args.variant
    if args.candidate:
        row=next((r for r in read('candidates/catalog.json') if r['id']==args.candidate),None)
        if row is None:raise ValueError('Unknown candidate ID')
        if variant and variant!=row['variant']:raise ValueError('Candidate/variant mismatch')
        variant=row['variant']; source=ROOT/row['path']; kind='candidate'
    elif args.source:
        source=Path(args.source).resolve();kind='external'
    else:
        if not variant:raise ValueError('--variant is required for preprocessed kernels')
        row=next(r for r in read('kernels/manifest.json') if r['variant']==variant)
        source=ROOT/row['path'];kind='preprocessed'
    if not variant:raise ValueError('--variant is required with --source')
    if kind!='external':
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256'],source
    level3=variant in ('gemm','trmm')
    commands=[]
    def execute(cmd):
        r=run_command(cmd,120);commands.append(r)
        (out/'build_commands.json').write_text(json.dumps(commands,indent=2))
        if r['returncode']!=0:raise RuntimeError('Compilation failed. See '+str(out/'build_commands.json'))
    version=run_command([args.cxx,'--version'],10)
    execute([args.cxx,*FLAGS,'-x','c++','-c',source,'-o',out/'kernel.o'])
    objects=[out/'kernel.o']
    if level3 and kind=='preprocessed':
        execute([args.cxx,*FLAGS,*(['-DTRMM_BENCH'] if variant=='trmm' else []),
                 '-c',ROOT/'benchmark/level3_wrapper.cpp','-o',out/'wrapper.o'])
        objects.append(out/'wrapper.o')
    driver=ROOT/'benchmark'/('level3.cpp' if level3 else 'level2.cpp')
    driver_flags=(['-DTRMM_BENCH'] if variant=='trmm' else []) if level3 else ['-DFAMILY='+str(VARIANTS.index(variant)-2)]
    execute([args.cxx,*FLAGS,*driver_flags,'-c',driver,'-o',out/'driver.o'])
    execute([args.cxx,out/'driver.o',*objects,'-lm','-o',out/'benchmark'])
    # An ABI-compatible scalar binary is not proof of RVV-profile admission.
    (out/'manifest.json').write_text(json.dumps(dict(variant=variant,kind=kind,
        candidate=args.candidate,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler=version,flags=FLAGS,profile_admission='not performed by numerical benchmark'),indent=2))
    (out/'inputs.json').write_text(json.dumps([c for c in read('configs/inputs.json') if c['variant']==variant],indent=2))
    print('Built '+str(out/'benchmark')+'. Transfer this output directory to physical RVV Linux for measurement.')

def measure(args):
    if platform.system()!='Linux' or platform.machine().lower() not in ('riscv64','riscv'):
        raise RuntimeError('Run measurement on physical RVV Linux, not on the compilation host.')
    if args.cpu is not None:os.sched_setaffinity(0,{args.cpu})
    folder=Path(args.build).resolve();manifest=json.loads((folder/'manifest.json').read_text())
    cases=json.loads((folder/'inputs.json').read_text())
    if args.smoke:cases=cases[:1]
    results=folder/'results.jsonl'
    if results.exists():raise FileExistsError('Preserve prior measurements. Use a fresh build directory.')
    failures=0
    for case in cases:
        for rep in range(args.repeats):
            seed=99101+rep+cases.index(case)*5
            av=['0',case['M'],case['N'],case['K'],seed,'0'] if manifest['variant'] in ('gemm','trmm') else [case['M'],case['N'],case['sx'],case['sy'],seed]
            r=run_command([folder/'benchmark',*map(str,av)],args.timeout)
            r.update(input_id=case['id'],replicate=rep,seed=seed)
            try:r['measurement']=json.loads(r.get('stdout','').strip().splitlines()[-1])
            except (ValueError,IndexError):r['measurement']={'status':'invalid_output'}
            passed=r['returncode']==0 and r['measurement']['status']=='pass'
            r['passed']=passed;failures+=not passed
            with results.open('a') as f:f.write(json.dumps(r)+'\n')
            print(case['id'],rep,'pass' if passed else 'FAIL',flush=True)
    (folder/'run_environment.json').write_text(json.dumps(dict(platform=platform.platform(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),repeats=args.repeats,smoke=args.smoke),indent=2))
    return int(failures>0)

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('verify');sub.add_parser('list')
    b=sub.add_parser('build');b.add_argument('--variant',choices=VARIANTS)
    group=b.add_mutually_exclusive_group();group.add_argument('--candidate');group.add_argument('--source')
    b.add_argument('--cxx',default='g++');b.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--build',required=True);r.add_argument('--cpu',type=int)
    r.add_argument('--smoke',action='store_true');r.add_argument('--repeats',type=int,default=3)
    r.add_argument('--timeout',type=float,default=120)
    args=ap.parse_args()
    if args.action=='verify':verify()
    elif args.action=='list':
        for row in read('candidates/catalog.json'):print(row['id'],row['variant'],row['profile'],row['organization'])
    elif args.action=='build':build(args)
    else:
        if args.repeats<1 or args.timeout<=0:ap.error('repeats and timeout must be positive')
        return measure(args)
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print('Error: '+str(exc),file=sys.stderr)
        sys.exit(1)

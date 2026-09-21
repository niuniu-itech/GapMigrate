"""Run construction, native search, frozen selection and held-out reports."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time
from .campaign import generate,VARIANTS
from .runtime import run_command
from .search import compile_driver,search_case,report_frozen,write_json,FLAGS

def validate_config(config):
    for key in ('search_budget_s','compile_s','audit_s','run_s'):
        if not math.isfinite(config[key]) or config[key]<=0:raise ValueError(key+' must be finite and positive')
    if not config['report_seeds']:raise ValueError('At least one held-out report seed is required')
    if len(set(config['report_seeds']))!=len(config['report_seeds']):raise ValueError('Report seeds must be distinct')
    for key in ('reference','qualification','timing'):
        if not math.isfinite(config['phase_limits'][key]) or config['phase_limits'][key]<=0:raise ValueError('Phase limits must be finite and positive')

def validate_profiles(profiles):
    if len({p['id'] for p in profiles})!=len(profiles):raise ValueError('Duplicate profiles')
    for p in profiles:
        if p['lmul'] not in (None,1):raise ValueError('Supported LMUL restriction is m1 or unrestricted')
        if not isinstance(p['reductions'],bool) or not isinstance(p['complex_memory'],bool):raise ValueError('Boolean profile flags required')

@contextlib.contextmanager
def measurement_core(cpu):
    import fcntl
    previous=os.sched_getaffinity(0)
    if cpu not in previous:raise ValueError('Selected CPU is outside the available affinity mask')
    lock=Path(tempfile.gettempdir())/f'gapmigrate-{os.getuid()}-cpu-{cpu}.lock'
    with lock.open('a') as f:
        # Avoid silently waiting while another timing job holds the same core.
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            os.sched_setaffinity(0,{cpu});yield
        finally:
            os.sched_setaffinity(0,previous);fcntl.flock(f,fcntl.LOCK_UN)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root',type=Path,default=Path('experiments/openblas'))
    p.add_argument('--config',type=Path,default=Path('configs/search.json'))
    p.add_argument('--candidate-dir',type=Path,help='Previously generated directory with catalog.json; omit to construct candidates')
    p.add_argument('--variant',choices=VARIANTS,action='append')
    p.add_argument('--profile',choices=['P0','P1','P2','P3','P4'],action='append')
    p.add_argument('--case',action='append',help='Registered input ID; repeat to select more than one')
    p.add_argument('--mode',choices=['pool','independent'],default='pool')
    p.add_argument('--plan-only',action='store_true',help='Construct and record jobs without hardware execution')
    p.add_argument('--cxx',default='g++');p.add_argument('--objdump',default='objdump')
    p.add_argument('--cpu',type=int)
    p.add_argument('--budget',type=float,help='Override search seconds per input and arm')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    config=json.loads(args.config.read_text());
    if args.budget is not None:config['search_budget_s']=args.budget
    validate_config(config)
    if not args.plan_only and (platform.system()!='Linux' or platform.machine().lower() not in ('riscv64','riscv')):
        p.error('Search and reports require physical RVV Linux. Use --plan-only for host-side construction.')
    root=args.benchmark_root.resolve();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty')
    all_profiles=json.loads((root/'configs/profiles.json').read_text());validate_profiles(all_profiles)
    profiles=[x for x in all_profiles if not args.profile or x['id'] in args.profile]
    cases=[x for x in json.loads((root/'configs/inputs.json').read_text())
           if (not args.variant or x['variant'] in args.variant) and (not args.case or x['id'] in args.case)]
    if not cases:p.error('No input configurations matched')
    if args.case and set(args.case)-{x['id'] for x in cases}:p.error('Some requested input IDs did not match the selected variants')
    # Selection uses per-case seeds; reports must use disjoint seeds.
    if {config['selection_seed']+int(c['id'].rsplit('_',1)[1]) for c in cases}&set(config['report_seeds']):
        p.error('Selection and held-out report seeds overlap')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    if args.candidate_dir:
        candidate_root=args.candidate_dir.resolve();catalog=json.loads((candidate_root/'catalog.json').read_text())
    else:
        candidate_root=out/'candidates'
        catalog,_=generate(root/'kernels',all_profiles,candidate_root,sorted({c['variant'] for c in cases}))
    catalog=[r for r in catalog if r['profile'] in {p['id'] for p in profiles} and r['variant'] in {c['variant'] for c in cases}]
    if any(not r.get('certificate',{}).get('protected_multiset_preserved') for r in catalog):
        p.error('Use a pool from generate_candidates.py with construction certificates, not the historical snapshot-only catalog')
    if any(not any(r['variant']==c['variant'] and r['profile']==profile['id'] for r in catalog) for c in cases for profile in profiles):
        p.error('Candidate pool does not cover every selected variant/profile')
    plan=dict(mode=args.mode,config=config,profiles=profiles,cases=cases,construction_s=time.monotonic()-started,
        compiler=args.cxx,objdump=args.objdump,flags=FLAGS,
        candidate_construction='source-derived generation' if not args.candidate_dir else 'existing candidate directory',
        search_scope='per-input shared profile pool; construction and common driver compilation excluded',
        comparison='observed-pool controls' if args.mode=='pool' else 'independent fixed/full searches with equal budgets',
        measurements='physical RVV execution required; plan is not a performance result')
    write_json(out/'plan.json',plan)
    if args.plan_only:
        print('Constructed candidates and wrote '+str(out/'plan.json')+'. No hardware measurement performed.');return 0
    cpu=args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    write_json(out/'environment.json',dict(platform=platform.platform(),cpu=cpu,
        compiler=run_command([args.cxx,'--version'],10),objdump=run_command([args.objdump,'--version'],10)))
    summary=[]
    with measurement_core(cpu):
        drivers={v:compile_driver(root/'benchmark',v,out/'drivers',args.cxx,config) for v in sorted({c['variant'] for c in cases})}
        for index,case in enumerate(cases):
            directory=out/'cases'/case['id'];directory.mkdir(parents=True)
            arms=['full'] if args.mode=='pool' else ['fixed','full']
            random.Random(config['order_seed']+index).shuffle(arms)
            mappings=[]
            for arm in arms:
                frozen=search_case(catalog,candidate_root,case,profiles,drivers[case['variant']],directory/arm,
                    arm,args.cxx,args.objdump,config)
                mappings.extend(dict(row,method='GapMigrate' if arm=='full' else 'Fixed organization') for row in frozen['choices'])
                if args.mode=='pool':
                    for control,rows in frozen.get('pool_controls',{}).items():
                        mappings.extend(dict(row,method='Pool fixed organization' if control=='fixed' else 'Pool fixed LMUL') for row in rows)
            write_json(directory/'report_mapping.json',mappings)
            result=report_frozen(case,mappings,directory/'reports',config)
            for row in result:
                row.update(input_id=case['id'],variant=case['variant'])
                # Export complete qualified source and a manifest for the existing benchmark CLI.
                if row['qualified']:
                    export=directory/'selected'/row['method'].replace(' ','_')/row['profile'];export.mkdir(parents=True)
                    shutil.copy2(row['source_path'],export/'kernel.cpp')
                    shutil.copy2(row['executable'],export/'benchmark')
                    write_json(export/'manifest.json',dict(variant=case['variant'],profile=row['profile'],candidate=row['candidate'],
                        source_sha256=row['source_sha256'],profile_admission='object audited with no unresolved external calls'))
                    write_json(export/'inputs.json',[case]);write_json(export/'selection.json',row)
            summary.extend(result);write_json(out/'summary.json',summary)
            print(case['id'],'qualified',sum(r['qualified'] for r in result),'of',len(result),flush=True)
    # Retain partial success and all failures, but do not signal total success if a method failed.
    return 0 if summary and all(r['qualified'] for r in summary) else 2

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as exc:
        print('Error: '+str(exc),file=sys.stderr);raise SystemExit(1)

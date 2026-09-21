"""Budgeted source-derived candidate search, object admission and frozen reports.

Pool controls and independent searches are explicit modes. Construction and
shared benchmark-driver compilation are recorded outside search budgets.
"""
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from .admission import audit
from .runtime import run_command,run_case,case_arguments

FLAGS=['-O3','-std=c++11','-march=rv64gcv_zvl256b','-mabi=lp64d',
       '-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize',
       '-fno-tree-slp-vectorize','-fno-tree-loop-distribute-patterns','-fno-builtin','-fno-stack-protector']

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')

def unique_candidates(catalog,variant,arm):
    """Equal source bytes share a trial, retaining every construction profile."""
    rank={'retain':0,'chunk_reduce':1,'regroup_4':1,'regroup_2':1,'materialize_vectors':2}
    rows=sorted([r for r in catalog if r['variant']==variant and (arm=='full' or r['organization']=='retain')],
                key=lambda r:(rank[r['organization']],r['lmul'],r['id']))
    unique={}
    for row in rows:
        if row['sha256'] not in unique:unique[row['sha256']]=dict(row,members=[])
        unique[row['sha256']]['members'].append(dict(id=row['id'],profile=row['profile'],organization=row['organization'],lmul=row['lmul']))
    return list(unique.values())

def select(pool,profiles,subset='full'):
    choices=[]
    for profile in profiles:
        pid=profile['id'];eligible=[]
        for row in pool:
            if pid not in row['admitted_profiles']:continue
            members=[m for m in row['members'] if m['profile']==pid]
            if subset=='fixed':members=[m for m in members if m['organization']=='retain']
            if subset=='fixed_lmul':members=[m for m in members if m['lmul']==(1 if pid in ('P1','P4') else 8)]
            if members:eligible.append((row,members[0]))
        best=min(eligible,key=lambda pair:pair[0]['measurement']['latency_us']) if eligible else None
        choices.append(dict(profile=pid,candidate=best[1]['id'] if best else None,
            executable=best[0]['executable'] if best else None,source_path=best[0]['source_path'] if best else None,
            source_sha256=best[0]['source_sha256'] if best else None,
            selection_latency_us=best[0]['measurement']['latency_us'] if best else None,
            eligible_candidates=len(eligible),status='selected' if best else 'no_eligible_implementation'))
    return choices

def compile_driver(benchmark,variant,directory,cxx,config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    level3=variant in ('gemm','trmm')
    flags=(['-DTRMM_BENCH'] if variant=='trmm' else []) if level3 else ['-DFAMILY='+str(['gemv_n','gemv_t','symv_L','symv_U'].index(variant))]
    source=Path(benchmark)/('level3.cpp' if level3 else 'level2.cpp');obj=directory/(variant+'.o')
    result=run_command([cxx,*FLAGS,*flags,'-c',source,'-o',obj],config['compile_s'])
    result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(directory/(variant+'_compile.json'),result)
    if result['status']!='pass':raise RuntimeError('Benchmark-driver compilation failed for '+variant)
    return obj

def search_case(catalog,candidate_root,case,profiles,driver,directory,arm,cxx,objdump,config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    candidates=unique_candidates(catalog,case['variant'],arm)
    started=time.monotonic();deadline=started+config['search_budget_s'];pool=[];tried=0
    choices=select([],profiles)
    def left(cap):return min(cap,deadline-time.monotonic())
    for candidate in candidates:
        if left(config['compile_s'])<=0:break
        tried+=1;job=directory/candidate['id'];job.mkdir()
        source=Path(candidate_root)/candidate['path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=candidate['sha256']:
            raise ValueError('Candidate checksum mismatch: '+candidate['id'])
        obj=job/'kernel.o';exe=job/'benchmark';phase='compile'
        details=[]
        def record(result):details.append(dict(phase=phase,**result));return result
        result=record(run_command([cxx,*FLAGS,'-c',source,'-o',obj],left(config['compile_s'])))
        admitted=[];gates=[]
        if result['status']=='pass':
            phase='object_admission';result=record(run_command([objdump,'-dr',obj],left(config['audit_s'])))
            if result['status']=='pass':
                (job/'disassembly.txt').write_text(result['stdout'])
                for profile in profiles:
                    gate=audit(result['stdout'],profile);gates.append(gate)
                    if gate['passed'] and not gate.get('external_call_symbols') and any(m['profile']==profile['id'] for m in candidate['members']):
                        admitted.append(profile['id'])
                if not admitted:result=dict(status='instruction_rejected')
        if result['status']=='pass':
            phase='link';result=record(run_command([cxx,driver,obj,'-lm','-o',exe],left(config['compile_s'])))
        if result['status']=='pass':
            phase='qualification_and_timing'
            seed=config['selection_seed']+int(case['id'].rsplit('_',1)[1])
            result=record(run_case([exe,*case_arguments(case,seed)],job/'selection',left(config['run_s']),config['phase_limits']))
        row=dict(candidate=candidate['id'],phase=phase,status=result['status'],details=details,gates=gates,
            members=candidate['members'],admitted_profiles=admitted,source_path=str(source.resolve()),
            source_sha256=candidate['sha256'],executable=str(exe.resolve()),elapsed_s=time.monotonic()-started)
        if result['status']=='pass' and time.monotonic()>=deadline:row['status']='late_result'
        if row['status']=='pass':
            row['measurement']=result['measurement'];pool.append(row)
        append(directory/'trials.jsonl',row)
        # Only successfully completed within-budget measurements enter selection.
        proposal=select(pool,profiles)
        if time.monotonic()<deadline:
            choices=proposal
            write_json(directory/'incumbent.json',dict(choices=choices,elapsed_s=time.monotonic()-started))
    frozen=dict(choices=choices,elapsed_s=time.monotonic()-started,arm=arm,budget_s=config['search_budget_s'],
        unique_candidates=len(candidates),tried=tried,measured=len(pool),
        timer_scope='candidate compilation, audit, linking, qualification, measurement and selection; construction and shared driver compilation excluded')
    if arm=='full':
        # These are pool controls, never labeled independent equal-budget arms.
        frozen['pool_controls']={}
        # Use only records present in the last committed pool, not post-deadline results.
        if (directory/'incumbent.json').exists():
            committed=json.loads((directory/'incumbent.json').read_text())['elapsed_s']
            committed_pool=[r for r in pool if r['elapsed_s']<=committed]
            frozen['pool_controls']['fixed']=select(committed_pool,profiles,'fixed')
            if case['variant'] not in ('gemm','trmm'):
                frozen['pool_controls']['fixed_lmul']=select(committed_pool,profiles,'fixed_lmul')
    write_json(directory/'frozen.json',frozen)
    return frozen

def report_frozen(case,mappings,directory,config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    jobs={}
    for row in mappings:
        if row.get('executable'):jobs.setdefault(row['executable'],[]).append({k:row[k] for k in ('method','profile','candidate')})
    measurements={exe:[] for exe in jobs};failed={exe:False for exe in jobs}
    for rep,seed in enumerate(config['report_seeds']):
        order=list(jobs);random.Random(seed).shuffle(order)
        for index,exe in enumerate(order):
            result=run_case([exe,*case_arguments(case,seed)],directory/f'rep_{rep}_{index}',config['run_s'],config['phase_limits'])
            append(directory/'reports.jsonl',dict(replicate=rep,seed=seed,executable=exe,
                shared_observation_mappings=jobs[exe],**result))
            if result['status']=='pass':measurements[exe].append(result['measurement']['latency_us'])
            else:failed[exe]=True
    summary=[]
    for row in mappings:
        exe=row.get('executable');values=measurements.get(exe,[])
        qualified=bool(exe) and not failed[exe] and len(values)==len(config['report_seeds'])
        summary.append(dict(row,qualified=qualified,report_processes=len(values),
            median_latency_us=statistics.median(values) if qualified else None,
            status='qualified' if qualified else 'no_eligible_implementation' if not exe else 'heldout_report_failed'))
    write_json(directory/'summary.json',summary)
    return summary

"""Bounded compiler processes and phase-aware physical-machine measurements."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

def terminate(process):
    if process.poll() is not None:return
    if os.name=='posix':
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    else:process.kill()
    process.wait()

def run_command(command,limit):
    command=[str(x) for x in command];start=time.monotonic()
    if limit<=0:return dict(status='deadline',command=command,latency=None)
    try:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,start_new_session=os.name=='posix')
    except OSError as exc:return dict(status='launch_failed',command=command,error=str(exc))
    try:
        out,err=process.communicate(timeout=limit)
        return dict(status='pass' if process.returncode==0 else 'failed',returncode=process.returncode,
                    stdout=out,stderr=err,command=command,elapsed_s=time.monotonic()-start)
    except subprocess.TimeoutExpired:
        terminate(process);out,err=process.communicate()
        return dict(status='timeout',stdout=out,stderr=err,command=command,latency=None,elapsed_s=time.monotonic()-start)

def run_case(command,folder,total_limit=85,phase_limits=None):
    """A timeout has no latency. Markers bound reference, validation and timing."""
    if total_limit<=0:return dict(status='deadline',latency=None)
    limits=phase_limits or dict(reference=60,qualification=15,timing=10)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();stage='reference';deadline=start+limits[stage];seen=0
    stdout=folder/'stdout.txt';stderr=folder/'stderr.txt'
    with stdout.open('w') as so,stderr.open('w') as se:
        try:process=subprocess.Popen([str(x) for x in command],stdout=so,stderr=se,start_new_session=os.name=='posix')
        except OSError as exc:return dict(status='launch_failed',error=str(exc),latency=None)
        while process.poll() is None:
            lines=stderr.read_text(errors='replace').splitlines()
            for line in lines[seen:]:
                if line in ('qualification_begin','timing_begin'):
                    stage=line.removesuffix('_begin');deadline=time.monotonic()+limits[stage]
            seen=len(lines)
            if time.monotonic()>min(deadline,start+total_limit):
                terminate(process)
                return dict(status=stage+'_timeout',stdout=stdout.read_text(errors='replace'),
                    stderr=stderr.read_text(errors='replace'),elapsed_s=time.monotonic()-start,latency=None)
            time.sleep(.01)
    result=dict(status='pass' if process.returncode==0 else stage+'_failed',returncode=process.returncode,
                stdout=stdout.read_text(errors='replace'),stderr=stderr.read_text(errors='replace'),elapsed_s=time.monotonic()-start)
    if result['status']=='pass':
        try:
            measurement=json.loads(result['stdout'].strip().splitlines()[-1])
            if measurement['status']!='pass':result['status']=measurement['status']
            elif not isinstance(measurement.get('latency_us'),(int,float)) or not math.isfinite(measurement['latency_us']) or measurement['latency_us']<=0:
                result['status']='invalid_latency'
            result['measurement']=measurement
        except (ValueError,IndexError,KeyError,TypeError):result['status']='invalid_output'
    return result

def case_arguments(case,seed):
    if case['variant'] in ('gemm','trmm'):
        return ['0',*[str(case[k]) for k in ('M','N','K')],str(seed),'0']
    return [*[str(case[k]) for k in ('M','N','sx','sy')],str(seed)]

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
from gapmigration.runtime import run_command,run_case
from gapmigration.search import select,unique_candidates,search_case,report_frozen

PROFILE=dict(id='P1',lmul=1,reductions=True,complex_memory=True)

def test_select_requires_construction_and_object_admission():
    row=dict(admitted_profiles=['P1'],members=[dict(id='good',profile='P1',organization='retain',lmul=1)],
        measurement=dict(latency_us=2),executable='good',source_path='s',source_sha256='h')
    unadmitted=dict(row,admitted_profiles=[],measurement=dict(latency_us=.01))
    other_profile=dict(row,members=[dict(id='bad',profile='P0',organization='retain',lmul=1)],measurement=dict(latency_us=.02))
    assert select([row,unadmitted,other_profile],[PROFILE])[0]['candidate']=='good'
    assert select([],[PROFILE])[0]['status']=='no_eligible_implementation'

def test_source_dedup_retains_profile_memberships():
    rows=[dict(id='a',sha256='same',variant='gemm',profile='P0',organization='retain',lmul=2),
          dict(id='b',sha256='same',variant='gemm',profile='P2',organization='retain',lmul=2)]
    result=unique_candidates(rows,'gemm','fixed')
    assert len(result)==1 and {x['profile'] for x in result[0]['members']}=={'P0','P2'}

def test_process_timeout_has_no_latency():
    row=run_command([sys.executable,'-c','import time; time.sleep(10)'],.1)
    assert row['status']=='timeout' and row['latency'] is None

@pytest.mark.parametrize('printed',["not-json",'{"status":"pass","latency_us":-1}', '{"status":"numerical_failed"}'])
def test_invalid_measurement_cannot_pass(tmp_path,printed):
    row=run_case([sys.executable,'-c','print('+repr(printed)+')'],tmp_path,5)
    assert row['status']!='pass'

def test_after_budget_measurement_not_selected(tmp_path):
    source=tmp_path/'source.cpp';source.write_text('void kernel(){}')
    rows=[dict(id='c1',variant='gemm',profile='P1',organization='retain',lmul=1,
               path=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest())]
    clock=[0.]
    def command(cmd,limit):
        clock[0]+=.1
        return dict(status='pass',stdout='0000 <kernel>:\n  0: 00007057 vsetvli a0,a1,e32,m1,ta,ma\n')
    def measurement(*args,**kw):
        clock[0]=3
        return dict(status='pass',measurement=dict(status='pass',latency_us=1))
    cfg=dict(search_budget_s=2,compile_s=1,audit_s=1,run_s=1,selection_seed=5100,phase_limits={})
    with patch('gapmigration.search.time.monotonic',side_effect=lambda:clock[0]),patch('gapmigration.search.run_command',side_effect=command),patch('gapmigration.search.run_case',side_effect=measurement):
        result=search_case(rows,tmp_path,dict(id='gemm_01',variant='gemm',M=1,N=1,K=1),[PROFILE],tmp_path/'driver.o',tmp_path/'search','full','g++','objdump',cfg)
    assert result['choices'][0]['candidate'] is None
    trial=json.loads((tmp_path/'search/trials.jsonl').read_text())
    assert trial['status']=='late_result'

def test_report_failure_retained_and_no_reselection(tmp_path):
    mappings=[dict(method='GapMigrate',profile='P1',candidate='chosen',executable='exe')]
    cfg=dict(report_seeds=[1,2],run_s=1,phase_limits={})
    responses=[dict(status='pass',measurement=dict(latency_us=2)),dict(status='numerical_failed')]
    with patch('gapmigration.search.run_case',side_effect=responses):
        result=report_frozen(dict(variant='gemm',M=1,N=1,K=1),mappings,tmp_path,cfg)
    assert not result[0]['qualified'] and result[0]['median_latency_us'] is None
    assert result[0]['candidate']=='chosen' and result[0]['report_processes']==1
    assert len((tmp_path/'reports.jsonl').read_text().splitlines())==2

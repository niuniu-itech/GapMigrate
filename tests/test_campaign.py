"""Regeneration is checked against historical artifacts, not self-generated fixtures."""
from pathlib import Path
import json
import re
import contextlib
import pytest
from gapmigration.campaign import generate
from gapmigration.frontend import Unsupported
from gapmigration.mapping.level2 import chunk_reduction
from gapmigration.mapping.materialize import construct as materialize
from gapmigration.scoped_packing import generate as packing
from gapmigration.migration import main as migrate

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'experiments/openblas'

def tokens(text):
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    text=re.sub(r'^\s*#include[^\n]*','',text,flags=re.M)
    return re.findall(r'\w+|[^\s]',text)

@pytest.fixture(scope='module')
def pool(tmp_path_factory):
    out=tmp_path_factory.mktemp('generated')
    rows,rejected=generate(BENCH/'kernels',json.loads((BENCH/'configs/profiles.json').read_text()),out)
    return out,rows,rejected

def test_all_109_historical_candidates_regenerated(pool):
    out,rows,rejected=pool;old=json.loads((BENCH/'candidates/catalog.json').read_text())
    assert len(rows)==109 and not rejected
    current={r['id']:r for r in rows}
    assert set(current)=={r['id'] for r in old}
    for row in old:
        new=current[row['id']]
        assert tokens((out/new['path']).read_text())==tokens((BENCH/row['path']).read_text()),row['id']
        assert new['certificate']['protected_multiset_preserved']
        assert new['status']=='constructed_not_qualified'

def test_untriggered_sources_are_retained(pool):
    _,rows,_=pool
    assert all(r['organization']=='retain' for r in rows if not r['certificate']['seed_nodes'])

def test_reduction_schema_mismatch_rejected():
    with pytest.raises(Unsupported,match='schema mismatch'):
        chunk_reduction('int source_kernel(void){return 0;}',1)

def test_live_workspace_materialization_rejected(pool):
    out,rows,_=pool;row=dict(next(r for r in rows if r['id']=='gemv_n_P3_m1_retain'))
    row['code']=(out/row['path']).read_text().replace('return 0;','buffer[0]=1; return 0;')
    with pytest.raises(Unsupported,match='workspace'):materialize(row)

@pytest.mark.parametrize('mode',['gather','tile'])
def test_packing_preserves_outer_ast(mode):
    source=BENCH/'auxiliary/packing.cpp'
    code,cert=packing(source,dict(id='P3',lmul=None,reductions=True,complex_memory=False),1,mode)
    assert cert['outside_region_ast_identical']
    assert cert['replaced_top_level_statement']>=0
    assert '__riscv_vlse32' not in code
    with pytest.raises(ValueError,match='LMUL gate'):
        packing(source,dict(id='P1',lmul=1,reductions=True,complex_memory=True),2,mode)

def test_host_plan_contains_no_measurements(pool,tmp_path):
    out,_,_=pool;dest=tmp_path/'plan'
    rc=migrate(['--benchmark-root',str(BENCH),'--config',str(ROOT/'configs/search.json'),
                '--candidate-dir',str(out),'--case','gemm_01','--profile','P1',
                '--mode','independent','--plan-only','--output',str(dest)])
    assert rc==0
    plan=json.loads((dest/'plan.json').read_text())
    assert plan['comparison']=='independent fixed/full searches with equal budgets'
    assert not (dest/'summary.json').exists()

def test_full_pipeline_wiring_with_mocked_hardware(pool,tmp_path,monkeypatch):
    """Exercise exports and phase wiring. This is not a hardware experiment."""
    import gapmigration.migration as pipeline
    import gapmigration.search as search
    out,_,_=pool;dest=tmp_path/'run'
    monkeypatch.setattr(pipeline.platform,'system',lambda:'Linux')
    monkeypatch.setattr(pipeline.platform,'machine',lambda:'riscv64')
    monkeypatch.setattr(pipeline.os,'sched_getaffinity',lambda _: {0},raising=False)
    monkeypatch.setattr(pipeline,'measurement_core',lambda _:contextlib.nullcontext())
    def command(cmd,limit):
        cmd=list(map(str,cmd))
        if '-o' in cmd:Path(cmd[cmd.index('-o')+1]).write_text('mock object/executable')
        return dict(status='pass',stdout='0000 <kernel>:\n  0: 00007057 vsetvli a0,a1,e32,m1,ta,ma\n')
    monkeypatch.setattr(pipeline,'run_command',command)
    monkeypatch.setattr(search,'run_command',command)
    monkeypatch.setattr(search,'run_case',lambda *a,**kw:dict(status='pass',measurement=dict(status='pass',latency_us=2)))
    rc=migrate(['--benchmark-root',str(BENCH),'--config',str(ROOT/'configs/search.json'),
                '--candidate-dir',str(out),'--case','gemm_01','--profile','P1',
                '--mode','independent','--output',str(dest)])
    assert rc==0
    summary=json.loads((dest/'summary.json').read_text())
    assert {r['method'] for r in summary}=={'GapMigrate','Fixed organization'}
    assert all(r['qualified'] and r['report_processes']==5 for r in summary)
    assert len(list(dest.glob('cases/gemm_01/selected/*/P1/kernel.cpp')))==2

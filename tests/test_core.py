import json
from pathlib import Path

import pytest

from gapmigration.admission import audit, self_test
from gapmigration.cli import PROFILES, candidates, main, read_function
from gapmigration.frontend import parse_function
from gapmigration.scope import UnsupportedControl, analyze

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/vector_scale.c"


def test_m1_scope_and_candidate(tmp_path):
    ast = read_function(EXAMPLE)
    scope, rows, rejected = candidates(ast, PROFILES["P1"])
    assert scope["seeds"]
    assert not scope["unsafe_crossings"]
    assert len(rows) == 1 and len(rejected) == 2
    assert "f32m2" not in rows[0]["code"]
    assert "f32m1" in rows[0]["code"]
    assert "i += vl" in rows[0]["code"]
    assert main(["candidates", str(EXAMPLE), "--output", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "manifest.json").read_text())["candidates"][0]["status"] == "constructed_not_validated"


def test_control_has_no_seed():
    scope, rows, _ = candidates(read_function(EXAMPLE), PROFILES["P0"])
    assert not scope["seeds"]
    assert "f32m2" in rows[0]["code"]


def test_closure_reaches_initialization_and_store():
    ast = parse_function('''void kernel(float*a,float*c,int n){
        vfloat32m2_t r=__riscv_vfmv_v_f_f32m2(0,4);
        for(int k=0;k<n;k++){r=__riscv_vfmacc_vf_f32m2(r,a[k],r,4);}
        __riscv_vse32_v_f32m2(c,r,4);
    }''')
    result = analyze(ast, PROFILES["P1"])
    selected = [n["text"] for n in result["nodes"] if n["id"] in result["selected"]]
    assert any("vfmv" in s for s in selected)
    assert any("vse32" in s for s in selected)
    assert not result["unsafe_crossings"]


def test_instruction_checks_fail_closed():
    self_test()
    assert not audit("", dict(PROFILES["P1"], id="P1"))["passed"]


def test_unstructured_control_is_rejected():
    with pytest.raises(UnsupportedControl):
        analyze(parse_function("void kernel(void){goto end;end:return;}"), PROFILES["P1"])

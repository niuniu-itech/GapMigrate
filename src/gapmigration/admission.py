"""Instruction-profile checks on disassembled kernel objects.

Objects must contain only kernels and their vector helpers. External calls are
reported separately and must be resolved/audited by the native runner.
"""
import collections, re

REDUCTION = re.compile(r'^v(?:f?w?red|fwred)')
COMPLEX_MEMORY = re.compile(r'^v[ls](?:s(?:e|seg)|[uo]x|seg)')
WHOLE_GROUP = re.compile(r'^(?:v[ls]([248])r|vmv([248])r)')

def instructions(text):
    function = None
    for line in text.splitlines():
        m = re.match(r'^\s*[0-9a-f]+ <([^>]+)>:', line)
        if m:
            function = m.group(1)
        m = re.match(r'^\s*([0-9a-f]+):\s+(?:(?:[0-9a-f]{16}|[0-9a-f]{8}|[0-9a-f]{4}|[0-9a-f]{2})\s+)+([a-z][a-z0-9_.]*)\s*(.*)$', line)
        if m:
            yield dict(address=m[1], mnemonic=m[2], operands=m[3], function=function)

def audit(text, profile):
    ins=list(instructions(text)); violations=[]; hist=collections.Counter(i['mnemonic'] for i in ins)
    if not ins:
        return dict(passed=False,violations=[dict(reason='no disassembled instructions')])
    for i in ins:
        op=i['mnemonic']; args=i['operands']; why=[]
        if not profile['reductions'] and REDUCTION.match(op):why.append('dedicated_vector_reduction')
        if not profile['complex_memory'] and COMPLEX_MEMORY.match(op):why.append('nonunit_or_segment_vector_memory')
        if profile['lmul']==1:
            if op in ('vsetvli','vsetivli') and not re.search(r'(?:^|,)m1(?:,|$)',args.replace(' ','')):why.append('LMUL_not_m1')
            if op=='vsetvl':why.append('dynamic_vtype_not_proven')
            if WHOLE_GROUP.match(op):why.append('whole_register_group_exceeds_one')
            if re.match(r'^v(?:fw(?:add|sub|mul|macc|msac|cvt)|w(?:add|sub|mul|macc|msac)|fncvt|ncvt|nsr|nclip)',op):why.append('widening_or_narrowing_EMUL_requires_review')
        for reason in why:violations.append(dict(**i,reason=reason))
    # R_RISCV relocations identify external calls before final linking; they are
    # resolved by the runner rather than assumed compliant.
    defined={i['function'] for i in ins}
    calls=sorted(set(re.findall(r'R_RISCV_CALL(?:_PLT)?\s+([^\s+]+)',text))-defined)
    return dict(passed=not violations,profile=profile['id'],instruction_count=len(ins),vector_instruction_count=sum(n for op,n in hist.items() if op.startswith('v')),histogram=dict(sorted(hist.items())),external_call_symbols=calls,violations=violations)

def self_test():
    profile=dict(id='P4',lmul=1,reductions=False,complex_memory=False)
    bad='''0000 <k>:
  0: 00007057 vsetvli a0,a1,e32,m8,ta,ma
  4: 00001057 vfredosum.vs v1,v2,v3
  8: 00001007 vlse32.v v1,(a0),a1
  c: 00001027 vsseg2e32.v v2,(a0)
 10: 00002057 vsetvl a0,a1,a2
'''
    result=audit(bad,profile)
    assert len(result['violations'])==5,result
    good=bad.splitlines()[0]+'\n  0: 00007057 vsetvli a0,a1,e32,m1,ta,ma\n  4: 00001007 vle32.v v1,(a0)\n'
    assert audit(good,profile)['passed']

if __name__=='__main__':self_test();print('legality negative/positive checks passed')

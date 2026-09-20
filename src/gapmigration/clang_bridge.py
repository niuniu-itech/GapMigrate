"""Optional Clang AST export; separate from the existing pycparser CFG path."""
import json
import subprocess
from pathlib import Path


def dump_ast(source, executable="clang", flags=(), timeout=60):
    source = Path(source)
    command = [executable, *flags, "-Xclang", "-ast-dump=json", "-fsyntax-only", str(source)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "Clang rejected this translation unit")
    return dict(frontend="clang", ast=json.loads(proc.stdout), diagnostics=proc.stderr,
                note="This export is not the input representation used by the prototype CFG analyzer.")

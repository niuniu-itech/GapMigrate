"""Small reproducible entrypoints around the recovered research implementation."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from .frontend import GEN, Unsupported, function_fragment, parse_function, preprocess, retarget_group
from .scope import CFG, analyze, unsupported
from .transforms import split_consumers

PROFILES = {
    "P0": dict(lmul=None, reductions=True, complex_memory=True),
    "P1": dict(lmul=1, reductions=True, complex_memory=True),
    "P2": dict(lmul=None, reductions=False, complex_memory=True),
    "P3": dict(lmul=None, reductions=True, complex_memory=False),
    "P4": dict(lmul=1, reductions=False, complex_memory=False),
}


def read_function(path, entry=None):
    text = Path(path).read_text(encoding="utf-8")
    if entry:
        text, _ = function_fragment(text, entry)
    return parse_function(preprocess(text, str(path)))


def candidates(ast, profile, old_lmul=2):
    """Enumerate supported FP32 retention/regrouping candidates; reject other gaps.

    Construction is NOT compilation, numerical qualification or optimization.
    Unrecognized grouping schemas never produce a claimed regrouped candidate.
    """
    scope = analyze(ast, profile)
    protected = Counter(n["text"] for n in scope["nodes"]
                        if n["id"] not in scope["selected"] and n["text"])
    choices = [(8, old_lmul)] if not scope["seeds"] else [(g, 1) for g in (8, 4, 2)]
    accepted, rejected = [], []
    for group, lmul in choices:
        name = f"g{group}_m{lmul}"
        try:
            node = copy.deepcopy(ast)
            events = []
            if group != 8:
                node, events = split_consumers(node, group)
                if not events:
                    raise Unsupported("source does not match the checked eight-consumer regrouping rule")
            node = retarget_group(node, old_lmul, lmul)
            nodes = CFG(node).nodes
            if any(unsupported(call, profile) for n in nodes for call in n["calls"]):
                raise Unsupported("remaining intrinsic gap has no rule in this CLI candidate path")
            remaining = Counter(n["text"] for n in nodes if n["text"])
            if protected - remaining:
                raise Unsupported("candidate changes statements outside the admitted region")
            code = "#include <stddef.h>\n#include <riscv_vector.h>\n" + GEN.visit(node) + "\n"
            accepted.append(dict(id=name, code=code, events=events,
                                 status="constructed_not_validated",
                                 sha256=hashlib.sha256(code.encode()).hexdigest()))
        except (Unsupported, AssertionError) as exc:
            rejected.append(dict(id=name, reason=str(exc)))
    return scope, accepted, rejected


def main(argv=None):
    parser = argparse.ArgumentParser(description="Source-aware vector kernel migration prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["analyze", "candidates"]:
        p = sub.add_parser(command)
        p.add_argument("source", type=Path)
        p.add_argument("--entry")
        p.add_argument("--profile", choices=PROFILES, default="P1")
        p.add_argument("--output", type=Path, required=True)
        if command == "candidates":
            p.add_argument("--source-lmul", type=int, choices=[1, 2, 4, 8], default=2)
    p = sub.add_parser("clang-ast", help="optional external Clang AST inspection")
    p.add_argument("source", type=Path)
    p.add_argument("--clang", default="clang")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--flag", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "clang-ast":
        from .clang_bridge import dump_ast
        result = dump_ast(args.source, args.clang, args.flag)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0
    ast = read_function(args.source, args.entry)
    profile = dict(PROFILES[args.profile], id=args.profile)
    if args.command == "analyze":
        if args.output.resolve() == args.source.resolve():
            parser.error("output must not overwrite source")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(analyze(ast, profile), indent=2), encoding="utf-8")
    else:
        scope, accepted, rejected = candidates(ast, profile, args.source_lmul)
        args.output.mkdir(parents=True, exist_ok=True)
        for row in accepted:
            (args.output / (row["id"] + ".c")).write_text(row.pop("code"), encoding="utf-8")
        result = dict(profile=profile, scope=scope, candidates=accepted, rejected=rejected,
                      qualification="requires compilation, object audit and runtime checks")
        (args.output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Constructed {len(accepted)} candidates; rejected {len(rejected)} unsupported choices.")
        return 0 if accepted else 2
    return 0

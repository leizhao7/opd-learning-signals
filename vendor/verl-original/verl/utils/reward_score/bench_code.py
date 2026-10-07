# EvalPlus-faithful scorer for HumanEval+ / MBPP+ inside verl's reward_score package.
# ground_truth = JSON: {"style": "humaneval"|"mbpp", "test": <evalplus test script>,
#                       "entry_point": <fn name>}
#
# Extraction is a faithful port of G-OPD's evalplus sanitizer
# (code_eval/coding/evalplus/evalplus/sanitize.py), which is what produced the
# paper's HE+/MBPP+ numbers: code_extract picks the longest syntactically valid
# line span, then a parse keeps top-level imports plus only the definitions
# reachable from entry_point (dependency closure). tree-sitter is unavailable
# on this box, so the parse uses Python's ast module; semantics mirrored
# 1:1 including the has-return-statement filter on plain functions.
#
# Execution: sanitized code + evalplus test script in a TEMP FILE (python -c
# hits ARG_MAX for the large "+" suites). exit 0 == pass. Module-level
# function => picklable for PrimeRewardManager's process pool.
import ast
import json
import os
import re
import subprocess
import sys
import tempfile


def _syntax_check(code):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError, ValueError, RecursionError):
        return False


def _span_search(lines):
    # evalplus code_extract: longest contiguous line span that parses, ranked
    # by number of non-empty lines.
    best, best_len = "", 0
    n = len(lines)
    for i in range(n):
        for j in range(i, n):
            chunk = "\n".join(lines[i : j + 1])
            if _syntax_check(chunk):
                cur = sum(1 for l in lines[i : j + 1] if l.strip())
                if cur > best_len:
                    best_len = cur
                    best = chunk
    return best


def _suffix_trim(lines):
    # cheap fallback for very long candidates: longest parsing prefix, then
    # longest parsing suffix of that prefix.
    n = len(lines)
    for j in range(n, 0, -1):
        chunk = "\n".join(lines[:j])
        if _syntax_check(chunk):
            for i in range(0, j):
                sub = "\n".join(lines[i:j])
                if _syntax_check(sub):
                    return sub
            return chunk
    return ""


def code_extract(text):
    # Fence lines are never valid Python, so valid spans cannot cross fenced
    # block boundaries: searching inside each fenced block (plus the whole
    # text when no fences) matches the global O(n^2) evalplus search at a
    # fraction of the cost.
    candidates = []
    parts = text.split("```")
    if len(parts) >= 2:
        for k, body in enumerate(parts):
            if k % 2 == 1:
                # fenced block: drop a leading language-tag line ("python",
                # "py", ...) — in the global evalplus search that tag sits on
                # the invalid ``` line and never enters a span.
                head, _, rest = body.partition("\n")
                if re.fullmatch(r"[A-Za-z0-9_+-]*\s*", head):
                    body = rest
            candidates.append(body)
    else:
        candidates.append(text)
    best, best_len = "", 0
    for cand in candidates:
        lines = cand.split("\n")
        if len(lines) > 250:
            got = "\n".join(lines) if _syntax_check(cand) else _suffix_trim(lines)
        else:
            got = cand if _syntax_check(cand) else _span_search(lines)
        cur = sum(1 for l in got.split("\n") if l.strip())
        if cur > best_len:
            best_len = cur
            best = got
    return best


def _collect_names(node):
    deps = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            deps.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            deps.add(sub.attr)
        elif isinstance(sub, ast.arg):
            deps.add(sub.arg)
    return deps


def _has_return(node):
    return any(isinstance(sub, ast.Return) for sub in ast.walk(node))


def _def_name(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def extract_target_code_or_empty(code, entrypoint=None):
    code = code_extract(code)
    try:
        tree = ast.parse(code)
    except (SyntaxError, MemoryError, ValueError, RecursionError):
        return ""
    lines = code.split("\n")

    def src(node):
        start = node.lineno
        for dec in getattr(node, "decorator_list", []) or []:
            start = min(start, dec.lineno)
        return "\n".join(lines[start - 1 : node.end_lineno])

    imports, defs, seen = [], [], set()
    for child in tree.body:
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            imports.append(child)
            continue
        name = _def_name(child)
        if name is None or name in seen:
            continue
        if isinstance(child, ast.FunctionDef) and not _has_return(child):
            # evalplus keeps only plain functions containing a return
            continue
        defs.append((name, child))
        seen.add(name)

    if entrypoint:
        name2deps = {name: _collect_names(node) for name, node in defs}
        reachable, queue = {entrypoint}, [entrypoint]
        while queue:
            cur = queue.pop(0)
            for dep in name2deps.get(cur, ()):
                if dep not in reachable:
                    reachable.add(dep)
                    queue.append(dep)

    out = []
    for node in imports:
        out.append(src(node))
    for name, node in defs:
        if entrypoint and name not in reachable:
            continue
        out.append(src(node))
    return "\n".join(out)


def sanitize(code, entrypoint=None):
    sanitized = extract_target_code_or_empty(code, entrypoint).strip()
    return sanitized if sanitized else code_extract(code)


def compute_score(solution_str, ground_truth, continuous=False):
    path = None
    try:
        gt = json.loads(ground_truth) if not isinstance(ground_truth, dict) else ground_truth
        # No entry_point (old-format rows) => sanitize without the dependency
        # closure (keeps all definitions) — guessing the target from the test
        # script misfires (e.g. picks isclose) and drops the real solution.
        code = sanitize(solution_str, gt.get("entry_point"))
        program = code + "\n\n" + gt["test"]
        if gt.get("style") == "humaneval":
            program += f"\n\ncheck({gt['entry_point']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
            path = f.name
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=300, env=env)
        return 1.0 if r.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

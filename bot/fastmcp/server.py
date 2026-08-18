"""FastMCP Server — StockTradingBot internal control plane.

All operations wrapped as MCP tools for seamless AI-agent interaction.
Runs via stdio (Claude/Cursor/OMP) or SSE HTTP endpoint.
"""

import ast
import asyncio
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from fastmcp import FastMCP
mcp = FastMCP(
    name="StockTradingBot",
    instructions=(
        "Internal control plane for the Stock Trading Bot codebase. "
        "Manages files, code search, git, strategy plugins, analytics, backtesting, "
        "and trading operations."
    ),
    version="0.1.0",
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRASH_DIR = PROJECT_ROOT / ".trash"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

for d in [LOGS_DIR, DATA_DIR, REPORTS_DIR, TRASH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_AUDIT_LOG = LOGS_DIR / "mcp_audit.log"


def _audit(action, params_hash, status):
    try:
        line = f"{datetime.now(timezone.utc).isoformat()} | {action} | {params_hash[:32]} | {status}\n"
        with open(_AUDIT_LOG, "a") as f:
            f.write(line)
    except Exception:
        pass


def _iter_py_files(root=None):
    root = root or PROJECT_ROOT
    files = []
    ignore_dirs = {"__pycache__", ".venv", ".git", ".trash"}
    ignore_globs = ["*.pyc"]
    for p in root.rglob("*.py"):
        if any(d in str(p) for d in ignore_dirs):
            continue
        if any(fnmatch.fnmatch(p.name, g) for g in ignore_globs):
            continue
        if "/test_" in str(p) and "tests" not in str(p):
            pass  # include test files
        files.append(p)
    return sorted(files)


# ════════════════════════════════════════════════════════════════════
# FILE MANAGEMENT TOOLS
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
def fs_read(path, start_line=None, end_line=None):
    """Read file contents with optional line range."""
    fp = PROJECT_ROOT / path
    if not fp.exists():
        raise FileNotFoundError(f"Not found: {path}")
    content = fp.read_text(encoding="utf-8", errors="replace")
    if start_line is not None or end_line is not None:
        lines = content.splitlines()
        s = (start_line - 1) if start_line else 0
        e = end_line if end_line else len(lines)
        content = "\n".join(lines[s:e])
    return f"# {path} ({len(content)} chars)\n{content}"


@mcp.tool()
def fs_write(path, content):
    """Create/overwrite file. Creates parent dirs automatically."""
    _audit("fs_write", path, "executing")
    fp = PROJECT_ROOT / path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8")
    _audit("fs_write", path, "ok")
    return f"Wrote {fp.relative_to(PROJECT_ROOT)} ({len(content)} bytes)"


@mcp.tool()
def fs_edit(path, before, after):
    """Replace exact text match. Fails if ambiguous (multiple matches)."""
    _audit("fs_edit", path, "executing")
    fp = PROJECT_ROOT / path
    if not fp.exists():
        raise FileNotFoundError(f"Not found: {path}")
    content = fp.read_text(encoding="utf-8", errors="replace")
    count = content.count(before)
    if count == 0:
        raise ValueError(f"'before' text not found in {path}")
    if count > 1:
        raise ValueError(f"'before' matches {count} times — use fs_write for bulk changes")
    new_content = content.replace(before, after, 1)
    fp.write_text(new_content, encoding="utf-8")
    _audit("fs_edit", path, "ok")
    return f"Replaced 1 occurrence in {path}"


@mcp.tool()
def fs_create_dir(path):
    """Create directory recursively."""
    fp = PROJECT_ROOT / path
    fp.mkdir(parents=True, exist_ok=True)
    return f"Created: {fp.relative_to(PROJECT_ROOT)}"


@mcp.tool()
def fs_list(path=".", show_hidden=False, max_depth=1):
    """List directory with types, sizes, extensions."""
    dp = PROJECT_ROOT / path
    if not dp.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    lines = [f"# Contents of {path}/"]
    excluded = {"__pycache__", ".venv", ".git"}
    items = sorted(dp.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    for item in items:
        if item.name.startswith(".") and not show_hidden:
            continue
        if item.name in excluded:
            continue
        sz = "?" if item.is_dir() else f"{item.stat().st_size:,}B"
        ext = item.suffix.lstrip(".").lower() or "(none)"
        kind = "DIR" if item.is_dir() else "FILE"
        lines.append(f"  {kind:>4} {item.name:<40} {sz:>10} .{ext}")
    return "\n".join(lines)


@mcp.tool()
def fs_search(pattern, target_path=".", glob_pattern="*.py", max_results=100):
    """Search file contents with regex. Returns matched lines with context."""
    from bot.fastmcp.utils.code_search import search_files as cs
    results = cs(pattern, paths=target_path, max_results=max_results)
    lines = [f"# Found {len(results)} match(es) for '{pattern}'"]
    for r in results[:max_results]:
        lines.append(f"\n**{r['file']}:{r['line']}** → `{r['match']}`")
        for _, ctx in r["context"]:
            lines.append(f"  > {ctx}")
    return "\n".join(lines)


@mcp.tool()
def fs_copy(source, destination):
    """Copy file or directory."""
    _audit("fs_copy", source, "executing")
    src = PROJECT_ROOT / source
    dst = PROJECT_ROOT / destination
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    _audit("fs_copy", source, "ok")
    return f"Copied {source} -> {destination}"


@mcp.tool()
def fs_move(source, destination):
    """Move/rename file or directory."""
    _audit("fs_move", source, "executing")
    src = PROJECT_ROOT / source
    dst = PROJECT_ROOT / destination
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    _audit("fs_move", source, "ok")
    return f"Moved {source} -> {destination}"


@mcp.tool()
def fs_delete(path):
    """Delete file — moves to .trash for recovery."""
    _audit("fs_delete", path, "executing")
    fp = PROJECT_ROOT / path
    if not fp.exists():
        raise FileNotFoundError(f"Not found: {path}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_name = f"{ts}_{fp.name}"
    (TRASH_DIR / trash_name).write_bytes(fp.read_bytes())
    fp.unlink()
    _audit("fs_delete", path, "ok")
    return f"Deleted {path} (recoverable from .trash/{trash_name})"


@mcp.tool()
def fs_info(path):
    """Get detailed file metadata: size, modified, hash, line count."""
    fp = PROJECT_ROOT / path
    if not fp.exists():
        raise FileNotFoundError(f"Not found: {path}")
    stat = fp.stat()
    import hashlib as _h
    h = _h.md5(fp.read_bytes()).hexdigest()
    content = fp.read_text(errors="replace")
    return json.dumps({
        "path": str(fp.relative_to(PROJECT_ROOT)),
        "is_file": fp.is_file(),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "md5": h,
        "line_count": len(content.splitlines()),
        "extension": fp.suffix.lstrip("."),
    }, indent=2)


@mcp.tool()
def fs_glob(pattern, root="."):
    """Glob pattern matching across filesystem."""
    rp = PROJECT_ROOT / root
    matches = list(rp.glob(pattern))
    rel = [str(m.relative_to(PROJECT_ROOT)) for m in matches]
    return json.dumps({"pattern": pattern, "matches": rel}, indent=2)


@mcp.tool()
def fs_project_structure():
    """Show current project tree respecting .gitignore."""
    from bot.fastmcp.utils.code_search import code_file_tree as cft
    tree = cft(depth=3)
    return json.dumps(tree, indent=2, default=str)


@mcp.tool()
def fs_watch(path="."):
    """Monitor directory for recent changes."""
    dp = PROJECT_ROOT / path
    if not dp.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    latest = None
    for p in dp.rglob("*"):
        try:
            mt = p.stat().st_mtime
            if latest is None or mt > latest[0]:
                latest = (mt, str(p.relative_to(PROJECT_ROOT)))
        except OSError:
            continue
    if latest:
        return json.dumps({"watch_path": path, "last_modified": latest[1],
                           "timestamp": datetime.fromtimestamp(latest[0]).isoformat()})
    return json.dumps({"watch_path": path, "message": "No files found"})


# ════════════════════════════════════════════════════════════════════
# CODE SEARCH & INDEXING TOOLS
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
def code_search(query, file_patterns=None, exclude_patterns=None, max_results=50):
    """Search codebase for patterns with 3-line context. Supports regex."""
    from bot.fastmcp.utils.code_search import search_files as cs
    results = cs(query, paths=file_patterns, exclude_patterns=exclude_patterns, max_results=max_results)
    lines = [f"# Found {len(results)} match(es) for '{query}'"]
    for r in results:
        lines.append(f"\n**{r['file']}:{r['line']}** → `{r['match']}`")
        for _, ctx in r["context"]:
            lines.append(f"  > {ctx}")
    return "\n".join(lines)


@mcp.tool()
def code_symbols(file_path=None, content=None):
    """Extract Python symbols (functions/classes/methods) using AST parser."""
    from bot.fastmcp.utils.code_search import extract_symbols
    symbols = extract_symbols(file_path=file_path, content=content)
    return json.dumps(symbols, indent=2, default=str)


@mcp.tool()
def code_find_references(symbol_name):
    """Find all references to a symbol across the codebase."""
    from bot.fastmcp.utils.code_search import find_references
    refs = find_references(symbol_name)
    return json.dumps(refs[:100], indent=2, default=str)


@mcp.tool()
def code_find_definitions(symbol_name):
    """Find where a symbol is defined in the codebase."""
    from bot.fastmcp.utils.code_search import find_definitions
    defs = find_definitions(symbol_name)
    return json.dumps(defs, indent=2, default=str)


@mcp.tool()
def code_dependency_graph(module_path):
    """Build import dependency graph for a module."""
    from bot.fastmcp.utils.code_search import dependency_graph
    deps = dependency_graph(module_path)
    return json.dumps(deps, indent=2, default=str)


@mcp.tool()
def code_classify_file(file_path):
    """Classify file purpose: test/module/config/plugin/entry-point/etc."""
    from bot.fastmcp.utils.code_search import class_file_type
    ft = class_file_type(file_path)
    return json.dumps({"file": file_path, "type": ft}, indent=2)


@mcp.tool()
def code_file_tree(max_depth=3, show_hidden=False):
    """Display project tree with depth limit, respecting .gitignore."""
    from bot.fastmcp.utils.code_search import code_file_tree as cft
    tree = cft(depth=max_depth, show_hidden=show_hidden)
    return json.dumps(tree, indent=2, default=str)


@mcp.tool()
def code_docstring_gen(file_path, function_name=None):
    """Generate Google-style docstrings for functions missing them."""
    fp = PROJECT_ROOT / file_path
    if not fp.exists():
        raise FileNotFoundError(f"Not found: {file_path}")
    source = fp.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    needs_update = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fname = node.name
            if function_name and fname != function_name:
                continue
            if not ast.get_docstring(node):
                params = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
                new_ds = '"""' + fname + ":\n\nArgs:\n" + "".join(f"    {p}: TODO\n" for p in params) + '"""'
                needs_update.append({"line": node.lineno, "function": fname, "new_docstring": new_ds})
    if not needs_update:
        return "No functions found without docstrings."
    return f"Generated {len(needs_update)} docstring(s) — apply with fs_edit on each function block."


@mcp.tool()
def code_rename(old_name, new_name, scope="."):
    """Rename a symbol everywhere using regex word-boundary replacement."""
    _audit("code_rename", f"{old_name}->{new_name}", "executing")
    root = PROJECT_ROOT / scope
    changed = []
    for py_path in _iter_py_files(root):
        try:
            src = py_path.read_text(encoding="utf-8", errors="replace")
            new_src = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, src)
            if new_src != src:
                py_path.write_text(new_src, encoding="utf-8")
                changed.append(str(py_path.relative_to(PROJECT_ROOT)))
        except Exception:
            continue
    _audit("code_rename", f"{old_name}->{new_name}", "ok")
    return f"Renamed '{old_name}' -> '{new_name}' in {len(changed)} file(s): {', '.join(changed[:10])}"


# ════════════════════════════════════════════════════════════════════
# GIT OPERATIONS TOOLS
# ════════════════════════════════════════════════════════════════════

def _git(*args):
    """Run git command and return stdout."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


@mcp.tool()
def git_status():
    """Working tree changes summary."""
    raw = _git("status", "--porcelain", "-uall")
    modified = []
    for line in raw.splitlines():
        sc, fp = line[:2].strip(), line[3:].strip('"')
        modified.append({"status": sc, "path": fp})
    detail = ""
    try:
        detail = _git("diff", "--stat")
    except Exception:
        pass
    return json.dumps({"modified": modified[:50], "summary": detail}, indent=2)


@mcp.tool()
def git_diff(staged=False):
    """Show detailed diff for staged/unstaged changes."""
    args = ["diff", "--cached"] if staged else ["diff"]
    return _git(*args)


@mcp.tool()
def git_add(paths=None):
    """Stage specific files or everything."""
    args = ["add"] + (paths if paths else ["."])
    _git(*args)
    return f"Staged: {len(paths) if paths else 'all'} path(s)"


@mcp.tool()
def git_commit(message):
    """Commit with message. Returns commit hash."""
    _audit("git_commit", message[:32], "executing")
    _git("commit", "-m", message)
    short_hash = _git("rev-parse", "--short", "HEAD")
    _audit("git_commit", message[:32], "ok")
    return f"Committed: {short_hash} — {message}"


@mcp.tool()
def git_push(remote="origin", branch="main"):
    """Push to remote repository."""
    _audit("git_push", f"{remote}/{branch}", "executing")
    out = _git("push", remote, branch)
    _audit("git_push", f"{remote}/{branch}", "ok")
    return out


@mcp.tool()
def git_pull(remote="origin", branch="main", strategy="ff-only"):
    """Pull latest changes (abort if conflicts)."""
    return _git("pull", remote, branch, f"--{strategy}")


@mcp.tool()
def git_log(n=20, grep=None, path=None):
    """Recent commits."""
    args = ["log", "--oneline", "-n", str(n)]
    if grep:
        args.extend(["--grep", grep])
    if path:
        args.extend(["--", path])
    return _git(*args)


@mcp.tool()
def git_branch(action="list", name=None):
    """List/create/delete branches."""
    if action == "list":
        return _git("branch", "-a")
    elif action == "create":
        if not name:
            raise ValueError("Branch name required")
        _audit("git_branch_create", name, "executing")
        out = _git("checkout", "-b", name)
        _audit("git_branch_create", name, "ok")
        return out
    elif action == "delete":
        if not name:
            raise ValueError("Branch name required")
        _audit("git_branch_delete", name, "executing")
        out = _git("branch", "-d", name)
        _audit("git_branch_delete", name, "ok")
        return out
    raise ValueError(f"Unknown action: {action}")


@mcp.tool()
def git_show(ref="HEAD"):
    """Show a specific commit."""
    return _git("show", ref, "--stat")


@mcp.tool()
def git_remote(action="list", url=None, name="origin"):
    """Manage remotes."""
    if action == "list":
        return _git("remote", "-v")
    elif action == "add":
        if not url:
            raise ValueError("URL required")
        return _git("remote", "add", name, url)
    elif action == "remove":
        return _git("remote", "remove", name)
    raise ValueError(f"Unknown action: {action}")


@mcp.tool()
def git_reset(mode="soft", ref="HEAD"):
    """Reset working tree/head. Modes: soft, mixed, hard."""
    allowed = {"soft", "mixed", "hard"}
    if mode not in allowed:
        raise ValueError(f"Mode must be one of {allowed}")
    _audit("git_reset", f"{mode}@{ref}", "executing")
    out = _git("reset", f"--{mode}", ref)
    _audit("git_reset", f"{mode}@{ref}", "ok")
    return out


@mcp.tool()
def git_revert(commit_ref, no_edit=True):
    """Safely revert a commit (creates new commit)."""
    _audit("git_revert", commit_ref[:12], "executing")
    args = ["revert"]
    if no_edit:
        args.append("--no-edit")
    args.append(commit_ref)
    out = _git(*args)
    _audit("git_revert", commit_ref[:12], "ok")
    return out


@mcp.tool()
def git_grep(pattern, pathspec=None, n_context=2):
    """Git-native grep with pathspec."""
    try:
        args = ["grep", "-n", "-C", str(n_context), pattern]
        if pathspec:
            args.extend(["--", pathspec])
        return _git(*args)
    except RuntimeError as e:
        if "no matches" in str(e):
            return "No matches found."
        raise


@mcp.tool()
def git_stash(action="save", message=None, pop=False):
    """Stash/pop/list/drop changes."""
    if action == "save":
        msg = f"-m {message}" if message else ""
        return _git("stash", "save", msg)
    elif action == "pop":
        return _git("stash", "pop")
    elif action == "list":
        return _git("stash", "list")
    elif action == "drop":
        return _git("stash", "drop")
    raise ValueError(f"Unknown stash action: {action}")


@mcp.tool()
def git_tags(list_only=True, name=None, annotated=False, message=None):
    """List/create tags."""
    if list_only:
        return _git("tag", "-l", "-n")
    elif name:
        if annotated and message:
            return _git("tag", "-a", name, "-m", message)
        return _git("tag", name)
    raise ValueError("Tag name required")


@mcp.tool()
def git_contribution_graph(author_filter=None, since=None):
    """Aggregate contribution stats from repo history."""
    if author_filter:
        args = ["shortlog", "-sne", "--author", author_filter]
    elif since:
        args = ["shortlog", "-sne", "--since", since]
    else:
        args = ["shortlog", "-sne"]
    raw = _git(*args)
    contributions = []
    for line in raw.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            contributions.append({"commits": int(parts[0]), "name": parts[-1], "email": parts[-2] if len(parts) > 2 else ""})
    return json.dumps({"contributions": contributions, "total_authors": len(contributions)}, indent=2)


@mcp.tool()
def git_verify():
    """Check git health: remote, branch, clean status."""
    return json.dumps({
        "repo_path": str(PROJECT_ROOT),
        "remote": _git("remote", "-v"),
        "branch": _git("branch", "-a"),
        "clean": len(_git("status", "--porcelain").strip()) == 0,
        "recent_commits": _git("log", "--oneline", "-5"),
    })


# ════════════════════════════════════════════════════════════════════
# PROJECT COMMANDS TOOLS
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
def project_install():
    """Install dependencies from requirements.txt."""
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        raise FileNotFoundError("requirements.txt not found")
    out = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                         capture_output=True, text=True)
    success = out.returncode == 0
    if not success:
        raise RuntimeError(out.stderr.strip()[:500])
    return f"Installed successfully.\n{out.stdout[-500:]}"


@mcp.tool()
def project_run_backtest(symbols="AAPL", start="2022-01-01", end=None, cash=100000, strategy="ema_cross_rsi"):
    """Run historical backtest and return metrics JSON."""
    _audit("project_run_backtest", f"{symbols}@{start}", "executing")
    cmd = [sys.executable, "main.py", "backtest", "--symbols", symbols, "--start", start]
    if end:
        cmd.extend(["--end", end])
    cmd.extend(["--cash", str(cash), "--strategy", strategy])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=180)
    output = result.stdout.strip()
    report_dir = REPORTS_DIR
    reports = list(report_dir.glob("*_backtest.html")) if report_dir.exists() else []
    _audit("project_run_backtest", f"{symbols}@{start}", "ok")
    return json.dumps({
        "output": output[-3000:],
        "reports_generated": [r.name for r in reports],
        "success": result.returncode == 0,
    })


@mcp.tool()
def project_run_dry_run(seconds=30):
    """Start dry-run engine for N seconds then stop cleanly."""
    _audit("project_run_dry_run", str(seconds), "executing")
    proc = subprocess.Popen([sys.executable, "main.py", "dry-run"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=str(PROJECT_ROOT))
    time.sleep(seconds)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    _audit("project_run_dry_run", str(seconds), "ok")
    return f"Ran dry-run for {seconds}s. Check logs/bot.log and logs/trades.csv for details."


@mcp.tool()
def project_test():
    """Run pytest test suite."""
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
                            capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    lines = result.stdout.splitlines()
    summary = [l for l in lines if l.startswith(("PASSED", "FAILED", "ERROR", "passed", "failed", "::"))]
    return json.dumps({
        "stdout": result.stdout[-2000:],
        "return_code": result.returncode,
        "summary_lines": summary[-20:],
        "success": result.returncode == 0,
    })


@mcp.tool()
def project_lint():
    """Check all Python source for syntax/import errors."""
    errors = []
    ok = 0
    for py_file in _iter_py_files():
        try:
            compile(py_file.read_text(encoding="utf-8", errors="replace"), str(py_file), "exec")
            ok += 1
        except SyntaxError as e:
            errors.append({"file": str(py_file.relative_to(PROJECT_ROOT)), "error": str(e)})
    return json.dumps({"checked": ok, "errors": errors, "healthy": len(errors) == 0})


@mcp.tool()
def project_send_order(symbol, qty, side="BUY", stop=None, target=None):
    """Place a trade through MockBroker (dry-run simulation)."""
    import asyncio
    from bot.broker import MockBroker
    broker = MockBroker()
    async def _trade():
        await broker.test_connection()
        oid = await broker.submit_order(symbol, qty, side, stop=stop, target=target)
        pos = await broker.get_positions()
        eq = await broker.get_equity()
        return {"order_id": oid, "position": pos, "equity": eq}
    result = asyncio.new_event_loop().run_until_complete(_trade())
    return json.dumps(result, indent=2)


@mcp.tool()
def project_check_env():
    """Validate environment: Python, packages, paths, git."""
    import pkg_resources
    checks = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "project_exists": PROJECT_ROOT.exists(),
    }
    # Check key packages
    for pkg in ["fastmcp", "mcp", "yfinance", "backtesting", "pandas", "numpy", "streamlit", "plotly", "vaderSentiment", "APScheduler", "requests"]:
        try:
            v = pkg_resources.get_distribution(pkg).version
            checks[f"{pkg}_installed"] = True
            checks[f"{pkg}_version"] = v
        except Exception:
            checks[f"{pkg}_installed"] = False
    # Git check
    try:
        checks["git_clean"] = len(_git("status", "--porcelain").strip()) == 0
    except Exception:
        checks["git_clean"] = "unknown"
    return json.dumps(checks, indent=2)


@mcp.tool()
def project_schema_update():
    """Refresh cached project metadata (strategies, datasources, plugins)."""
    from bot.core.plugins import discover_all
    from bot.core import STRATEGIES, DATASOURCES, SENTIMENT_SOURCES
    discover_all()
    meta = {
        "strategies": [{"name": n, "params": STRATEGIES.get(n).params if hasattr(STRATEGIES.get(n), "params") else {}} for n in STRATEGIES.names()],
        "datasources": [{"name": n, "priority": getattr(DATASOURCES.get(n), "priority", "N/A")} for n in DATASOURCES.names()],
        "sentiment_sources": [{"name": n} for n in SENTIMENT_SOURCES.names()],
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "version": 1,
    }
    from bot.fastmcp.schemas import save_project_metadata
    save_project_metadata(meta)
    return json.dumps(meta, indent=2)


@mcp.tool()
def project_send_sentiment(tickers, hours=24):
    """Quick sentiment analysis for ticker(s)."""
    from bot.sentiment import SentimentEngine
    engine = SentimentEngine()
    results = {}
    for sym in tickers:
        score = engine.score(sym, hours=hours)
        results[sym] = {
            "mentions": score.mentions,
            "bullish_pct": round(score.bullish / max(score.mentions, 1) * 100, 1),
            "bearish_pct": round(score.bearish / max(score.mentions, 1) * 100, 1),
            "net_score": score.net_score,
        }
    return json.dumps(results, indent=2)


@mcp.tool()
def project_help():
    """List all available project commands."""
    return """Project Commands:
  project_install          Install dependencies
  project_run_backtest     Run historical backtest
  project_run_dry_run      Start engine (MockBroker) for N seconds
  project_test             Run pytest
  project_lint             Syntax check all source
  project_send_order       Place mock order via MockBroker
  project_check_env        Validate environment
  project_schema_update    Refresh plugin metadata cache
  project_send_sentiment   Quick sentiment analysis
"""


# ════════════════════════════════════════════════════════════════════
# ANALYTICS & PORTFOLIO TOOLS
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
def analytics_trade_journal(symbol=None, date_from=None, date_to=None, side=None, reason=None, limit=100):
    """Query trades CSV with filters."""
    from bot.fastmcp.utils.analytics import query_trades
    return json.dumps(query_trades(symbol=symbol, date_from=date_from, date_to=date_to,
                                   side=side, reason=reason, limit=limit), indent=2)


@mcp.tool()
def analytics_portfolio_summary():
    """Current positions + equity snapshot."""
    from bot.fastmcp.utils.analytics import portfolio_summary
    return json.dumps(portfolio_summary(), indent=2)


@mcp.tool()
def analytics_equity_curve_stats():
    """Equity curve summary: start/end/sharpe/maxDD/gain."""
    from bot.fastmcp.utils.analytics import equity_curve_stats
    return json.dumps(equity_curve_stats(), indent=2)


@mcp.tool()
def analytics_daily_pnl(window_days=14):
    """Daily P&L breakdown grouped by day."""
    from bot.fastmcp.utils.analytics import daily_pnl_breakdown
    return json.dumps(daily_pnl_breakdown(window_days=window_days), indent=2)


@mcp.tool()
def analytics_risk_metrics():
    """Risk metrics: Sharpe, Sortino, Calmar, max consecutive losses."""
    from bot.fastmcp.utils.analytics import risk_metrics
    return json.dumps(risk_metrics(), indent=2)


@mcp.tool()
def analytics_kill_switch_stats():
    """Kill switch tripping frequency and conditions."""
    from bot.fastmcp.utils.analytics import kill_switch_stats
    return json.dumps(kill_switch_stats(), indent=2)


@mcp.tool()
def analytics_strategy_compare(strategy_a, strategy_b, start="2022-01-01", symbols="AAPL"):
    """Compare two strategies head-to-head."""
    from bot.fastmcp.utils.strategies import strategy_compare
    syms = [s.strip() for s in symbols.split(",")]
    result = strategy_compare(strategy_a, strategy_b, start, syms)
    return json.dumps(result, indent=2)


@mcp.tool()
def analytics_backtest_report():
    """Read last backtest results from reports/."""
    reports = list(REPORTS_DIR.glob("*_backtest.html")) if REPORTS_DIR.exists() else []
    summaries = list((LOGS_DIR / "backtest_summary.json")).exists() and json.loads((LOGS_DIR / "backtest_summary.json").read_text()) if (LOGS_DIR / "backtest_summary.json").exists() else {}
    return json.dumps({
        "html_reports": [r.name for r in reports[:5]],
        "summary": summaries,
        "latest_timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ════════════════════════════════════════════════════════════════════
# STRATEGY LIFECYCLE TOOLS
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
def strategy_list():
    """List all registered strategies with parameters."""
    from bot.fastmcp.utils.strategies import list_strategies
    return json.dumps(list_strategies(), indent=2)


@mcp.tool()
def strategy_create(name, description="", base_class="EmaCrossRsi"):
    """Scaffold a new strategy plugin template."""
    from bot.fastmcp.utils.strategies import scaffold_strategy
    return scaffold_strategy(name, description, base_class)


@mcp.tool()
def strategy_validate(name, trends="uptrend,downtrend,sideways,volatile"):
    """Test-validate strategy against synthetic data trends."""
    from bot.fastmcp.utils.strategies import validate_strategy
    t = [t.strip() for t in trends.split(",")]
    return json.dumps(validate_strategy(name, t), indent=2)


@mcp.tool()
def strategy_params_doc(name=None):
    """Generate documentation for strategy parameter schema."""
    from bot.core import STRATEGIES
    discover_all = __import__("bot.core.plugins", fromlist=["discover_all"]).discover_all
    discover_all()
    targets = [name] if name else STRATEGIES.names()
    docs = {}
    for sname in targets:
        s = STRATEGIES.get(sname)
        params = getattr(s, "params", {})
        docs[sname] = {"params": params, "has_signals": callable(getattr(s, "generate_signals", None))}
    return json.dumps(docs, indent=2)


@mcp.tool()
def strategy_optimize(name, param_ranges, start="2022-01-01", cash=100000):
    """Grid-search strategy parameters via backtesting."""
    from bot.fastmcp.utils.strategies import optimize_strategy_params
    ranges = json.loads(param_ranges) if isinstance(param_ranges, str) else param_ranges
    result = optimize_strategy_params(name, ranges, start, cash)
    return json.dumps(result, indent=2)


# ════════════════════════════════════════════════════════════════════
# RESOURCE HELPERS
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
def resource_file(path):
    """Read file as resource (returns raw content for MCP clients)."""
    fp = PROJECT_ROOT / path
    if not fp.exists():
        return json.dumps({"error": f"File not found: {path}"})
    return fp.read_text(encoding="utf-8", errors="replace")


@mcp.tool()
def resource_log_tail(name="bot.log", tail=200):
    """Read last N lines of a log file."""
    lp = LOGS_DIR / name
    if not lp.exists():
        return json.dumps({"error": f"Log not found: {name}"})
    lines = lp.read_text().splitlines()[-tail:]
    return "\n".join(lines)


@mcp.tool()
def resource_settings():
    """Return current Settings serialized to JSON."""
    from bot.config import load_settings
    s = load_settings()
    d = {k: str(v) if k == "robinhood_mcp_auth_header" else v for k, v in s.__dict__.items()}
    d["auth_header_masked"] = "***" + d.get("robinhood_mcp_auth_header", "")[-4:] if d.get("robinhood_mcp_auth_header") else ""
    del d["robinhood_mcp_auth_header"]
    return json.dumps(d, indent=2)


@mcp.tool()
def resource_meta_project():
    """Cached project metadata (strategies, datasources, etc.)."""
    from bot.fastmcp.schemas import load_project_metadata
    return json.dumps(load_project_metadata(), indent=2)


# ════════════════════════════════════════════════════════════════════
# HELPER TOOLS
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
def todo_list(action="list", task=None, phase=None, reason=None):
    """CRUD operation on TODO items: list/add/complete/drop/block/unblock/remove."""
    actions_allowed = ["list", "add", "complete", "drop", "block", "unblock", "remove", "init", "view"]
    if action not in actions_allowed:
        raise ValueError(f"Action must be one of {actions_allowed}")
    
    if action == "list":
        return "todo view"  # Delegate to OMP's own todo
    
    if action == "add":
        if not task:
            raise ValueError("Task string required")
        return f"todo add: {task}"  # Delegate
    
    if action == "init":
        if not task:
            raise ValueError("JSON list required for init")
        return f"todo init: {task}"
    
    if action == "complete":
        return f"todo done: {task}" if task else "todo done: {phase}"
    
    if action == "drop":
        return f"todo drop: {task}" if task else "todo drop: {phase}"
    
    if action == "block":
        return f"todo block: {task} reason={reason}" if task else f"todo block: {phase} reason={reason}"
    
    if action == "unblock":
        return f"todo unblock: {task}" if task else "todo unblock: {phase}"
    
    if action == "remove":
        return f"todo rm: {task}" if task else "todo rm: {phase}"
    
    if action == "view":
        return "todo view"
    
    return json.dumps({"message": f"Use action: {action}"})


@mcp.tool()
def env_var_get(key=None):
    """Get or set runtime environment variables."""
    if key:
        val = os.getenv(key)
        return json.dumps({"key": key, "value": val})
    # List all USER-level vars
    user_vars = {k: v for k, v in os.environ.items() if not k.startswith("_")}
    return json.dumps(user_vars, indent=2)


@mcp.tool()
def env_var_set(key, value):
    """Set a session-scoped environment variable."""
    os.environ[key] = value
    return f"Set {key}='{value}'"


@mcp.tool()
def time_now(tz=None):
    """Get current system time with optional timezone offset."""
    now = datetime.now(timezone.utc)
    if tz and isinstance(tz, int):
        from datetime import timedelta as td
        now = now.replace(tzinfo=timezone(td(hours=tz)))
    return json.dumps({"utc": now.isoformat(), "human": now.strftime("%Y-%m-%d %H:%M:%S UTC")})


@mcp.tool()
def http_request(method="GET", url=None, headers=None, body=None, timeout=10):
    """Make outbound HTTP request. For external APIs/data sources."""
    import requests as req_lib
    hdrs = json.loads(headers) if isinstance(headers, str) else headers
    bdy = json.loads(body) if isinstance(body, str) else body
    resp = req_lib.request(method, url, headers=hdrs, json=bdy, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = resp.text[:2000]
    return json.dumps({"status_code": resp.status_code, "headers": dict(resp.headers), "body": data}, indent=2)


@mcp.tool()
def web_crawl(url, extract_links=True, max_chars=5000):
    """Crawl a URL and extract text content."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "StockTradingBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        
        # Strip HTML tags
        import re
        text = re.sub(r'<[^>]+>', '\n', content)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        result = {"url": url, "raw_preview": text[:max_chars]}
        if extract_links:
            links = re.findall(r'href=[\"\'](.*?)[\"\']', content)
            result["links_found"] = len(links)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"url": url, "error": str(e)})


@mcp.tool()
def help_topics():
    """List all available tool categories and their names."""
    return """
Available Tool Categories:

FILE MANAGEMENT (12):
  fs_read, fs_write, fs_edit, fs_create_dir, fs_list, fs_search,
  fs_copy, fs_move, fs_delete, fs_info, fs_glob, fs_project_structure,
  fs_watch

CODE SEARCH (9):
  code_search, code_symbols, code_find_references, code_find_definitions,
  code_dependency_graph, code_classify_file, code_file_tree,
  code_docstring_gen, code_rename

GIT OPERATIONS (16):
  git_status, git_diff, git_add, git_commit, git_push, git_pull,
  git_log, git_branch, git_show, git_remote, git_reset, git_revert,
  git_grep, git_stash, git_tags, git_contribution_graph, git_verify

PROJECT COMMANDS (11):
  project_install, project_run_backtest, project_run_dry_run,
  project_test, project_lint, project_send_order, project_check_env,
  project_schema_update, project_send_sentiment, project_help

ANALYTICS (8):
  analytics_trade_journal, analytics_portfolio_summary,
  analytics_equity_curve_stats, analytics_daily_pnl,
  analytics_risk_metrics, analytics_kill_switch_stats,
  analytics_strategy_compare, analytics_backtest_report

STRATEGY LIFECYCLE (6):
  strategy_list, strategy_create, strategy_validate,
  strategy_params_doc, strategy_optimize

RESOURCES (4):
  resource_file, resource_log_tail, resource_settings, resource_meta_project

HELPERS (8):
  todo_list, env_var_get, env_var_set, time_now,
  http_request, web_crawl, help_topics

TOTAL: ~70 tools
"""


@mcp.tool()
def help_tool(tool_name):
    """Get details about a specific tool (description, purpose)."""
    desc_map = {
        "fs_read": "Read file contents with optional line range.",
        "fs_write": "Create/overwrite file with auto-parent-creation.",
        "fs_edit": "Replace exact text match (fails if ambiguous).",
        "fs_create_dir": "Create directory recursively.",
        "fs_list": "List directory with types, sizes, extensions.",
        "fs_search": "Regex search with context lines across files.",
        "fs_copy": "Copy file or directory.",
        "fs_move": "Move/rename file or directory.",
        "fs_delete": "Soft-delete to .trash (recoverable).",
        "fs_info": "Detailed file metadata: size, hash, modified.",
        "fs_glob": "Glob pattern matching.",
        "fs_project_structure": "Project tree respecting .gitignore.",
        "fs_watch": "Monitor directory for recent modifications.",
        "code_search": "Text/regex search with surrounding context.",
        "code_symbols": "AST-based symbol extraction.",
        "code_find_references": "Find all usages of a symbol.",
        "code_find_definitions": "Locate symbol definitions.",
        "code_dependency_graph": "Import dependency map for a module.",
        "code_classify_file": "Classify file purpose (test/mock/etc).",
        "code_file_tree": "Project tree visualization.",
        "code_docstring_gen": "Auto-generate missing docstrings.",
        "code_rename": "Safe cross-file symbol rename.",
        "git_status": "Working tree changes summary.",
        "git_diff": "Diff content (staged or unstaged).",
        "git_add": "Stage files.",
        "git_commit": "Commit with message.",
        "git_push": "Push to remote.",
        "git_pull": "Pull latest (ff-only).",
        "git_log": "Recent commits.",
        "git_branch": "List/create/delete branches.",
        "git_show": "Show specific commit.",
        "git_remote": "Manage remotes.",
        "git_reset": "Reset working tree (soft/mixed/hard).",
        "git_revert": "Safely revert a commit.",
        "git_grep": "Git-native grep.",
        "git_stash": "Stash/pop/list/drop changes.",
        "git_tags": "List/create tags.",
        "git_contribution_graph": "Author contribution statistics.",
        "git_verify": "Git health check.",
        "project_install": "Install pip dependencies.",
        "project_run_backtest": "Historical backtest execution.",
        "project_run_dry_run": "Start engine with MockBroker.",
        "project_test": "Run pytest suite.",
        "project_lint": "Syntax check all Python files.",
        "project_send_order": "Place mock order via MockBroker.",
        "project_check_env": "Environment validation.",
        "project_schema_update": "Refresh plugin metadata cache.",
        "project_send_sentiment": "Quick social sentiment analysis.",
        "analytics_trade_journal": "Query trades CSV.",
        "analytics_portfolio_summary": "Position + equity snapshot.",
        "analytics_equity_curve_stats": "Equity curve statistics.",
        "analytics_daily_pnl": "Daily profit/loss breakdown.",
        "analytics_risk_metrics": "Sharpe/Sortino/Calmar ratios.",
        "analytics_kill_switch_stats": "Emergency stop stats.",
        "analytics_strategy_compare": "Head-to-head strategy comparison.",
        "analytics_backtest_report": "Latest backtest results.",
        "strategy_list": "List registered strategies.",
        "strategy_create": "Scaffold new strategy plugin.",
        "strategy_validate": "Test strategy against synthetic trends.",
        "strategy_params_doc": "Strategy parameter documentation.",
        "strategy_optimize": "Grid-search best parameters.",
        "resource_file": "Read file as MCP resource.",
        "resource_log_tail": "Last N lines of a log.",
        "resource_settings": "Current settings as JSON.",
        "resource_meta_project": "Cached project metadata.",
        "todo_list": "TODO CRUD operations.",
        "env_var_get": "Get environment variables.",
        "env_var_set": "Set environment variables.",
        "time_now": "Current timestamp.",
        "http_request": "Outbound HTTP request.",
        "web_crawl": "Crawl URL for content.",
        "help_topics": "List all tool categories.",
        "help_tool": "Describe specific tool.",
    }
    return json.dumps({"tool": tool_name, "description": desc_map.get(tool_name, "No description available.")})


@mcp.tool()
def note(text):
    """Quick note-taking. Append to scratch pad."""
    notes_file = PROJECT_ROOT / "logs" / "mcp_scratch_notes.txt"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(notes_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n\n")
    return f"Note saved: {notes_file}"


# ════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")

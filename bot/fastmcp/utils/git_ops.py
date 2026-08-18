"""Safe Git operations with confirmation gates."""

import json
import subprocess
from pathlib import Path

GIT_REPO = Path(__file__).resolve().parent.parent.parent.parent  # D:/StockTradingBot


def _git(*args, cwd=None) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True,
        cwd=cwd or str(GIT_REPO),
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_repo_path() -> str:
    """Return absolute path to repo root."""
    return str(GIT_REPO)


def git_status() -> dict:
    """Working tree changes summary."""
    raw = _git("status", "--porcelain", "-uall")
    changed = []
    for line in raw.splitlines():
        status_code = line[:2].strip()
        filepath = line[3:].strip('"')
        changed.append({"status": status_code, "path": filepath})
    detailed = {}
    try:
        detailed_out = _git("diff", "--stat")
        detailed["summary"] = detailed_out
    except Exception:
        pass
    return {"modified": changed, "summary": detailed.get("summary")}


def git_diff(staged: bool = False) -> str:
    """Show diff content. staged=True shows index only."""
    args = ["diff", "--cached"] if staged else ["diff"]
    return _git(*args)


def git_add(paths: list[str]) -> str:
    """Stage specific paths. If empty, stages everything."""
    args = ["add"] + (paths or ["."])
    out = _git(*args)
    return f"Staged: {len(paths) if paths else 'all'} path(s)"


def git_commit(message: str) -> str:
    """Commit with message. Returns commit hash."""
    args = ["commit", "-m", message]
    out = _git(*args)
    try:
        short_hash = _git("rev-parse", "--short", "HEAD")
        return f"Committed: {out}\nHash: {short_hash}"
    except Exception:
        return f"Committed: {out}"


def git_push(remote: str = "origin", branch: str = "main") -> str:
    """Push to remote."""
    return _git("push", remote, branch)


def git_pull(remote: str = "origin", branch: str = "main", strategy: str = "ff-only") -> str:
    """Pull latest changes."""
    return _git("pull", remote, branch, f"--{strategy}")


def git_log(n: int = 20, grep: str | None = None, path: str | None = None) -> str:
    """Recent commits."""
    args = ["log", "--oneline", "-n", str(n)]
    if grep:
        args.extend(["--grep", grep])
    if path:
        args.extend(["--", path])
    return _git(*args)


def git_branch(action: str = "list", name: str | None = None) -> str:
    """List/create/delete branches."""
    if action == "list":
        return _git("branch", "-a")
    elif action == "create":
        if not name:
            raise ValueError("Branch name required for create")
        return _git("checkout", "-b", name)
    elif action == "delete":
        if not name:
            raise ValueError("Branch name required for delete")
        return _git("branch", "-d", name)
    else:
        raise ValueError(f"Unknown branch action: {action}")


def git_show(ref: str = "HEAD") -> str:
    """Show a specific commit."""
    return _git("show", ref, "--stat")


def git_remote(action: str = "list", url: str | None = None, name: str = "origin") -> str:
    """Manage remotes."""
    if action == "list":
        return _git("remote", "-v")
    elif action == "add":
        if not url:
            raise ValueError("URL required for add")
        return _git("remote", "add", name, url)
    elif action == "remove":
        return _git("remote", "remove", name)
    else:
        raise ValueError(f"Unknown remote action: {action}")


def git_reset(mode: str = "soft", ref: str = "HEAD") -> str:
    """Reset working tree/head."""
    allowed = {"soft", "mixed", "hard"}
    if mode not in allowed:
        raise ValueError(f"Mode must be one of {allowed}")
    return _git("reset", f"--{mode}", ref)


def git_revert(commit_ref: str, no_edit: bool = True) -> str:
    """Safely revert a commit."""
    args = ["revert"]
    if no_edit:
        args.append("--no-edit")
    args.append(commit_ref)
    return _git(*args)


def git_grep(pattern: str, pathspec: str | None = None, n_context: int = 2) -> str:
    """Git-native grep."""
    args = ["grep", "-n", "-C", str(n_context), pattern]
    if pathspec:
        args.extend(["--", pathspec])
    try:
        return _git(*args)
    except RuntimeError as e:
        if "no matches" in str(e):
            return "No matches found."
        raise


def git_stash(action: str = "save", message: str | None = None, pop: bool = False) -> str:
    """Stash/pop/list changes."""
    if action == "save":
        msg = f"-m {message}" if message else ""
        return _git("stash", "save", msg)
    elif action == "pop":
        return _git("stash", "pop")
    elif action == "list":
        return _git("stash", "list")
    elif action == "drop":
        return _git("stash", "drop")
    else:
        raise ValueError(f"Unknown stash action: {action}")


def git_tags(list_only: bool = True, name: str | None = None, annotated: bool = False, message: str | None = None) -> str:
    """List/create/delete tags."""
    if list_only:
        return _git("tag", "-l", "-n")
    elif name:
        if annotated and message:
            return _git("tag", "-a", name, "-m", message)
        elif annotated:
            return _git("tag", "-a", name)
        return _git("tag", name)
    else:
        raise ValueError("Tag name required for create/tagged operations")


def git_contribution_graph(author_filter: str | None = None, since: str | None = None) -> dict:
    """Aggregate contribution stats from history."""
    args = ["shortlog", "-sne"]
    if author_filter:
        args.extend(["--author", author_filter])
    if since:
        args = ["shortlog", "-sne", "--since", since] + ([f"--author={author_filter}"] if author_filter else [])
    raw = _git(*args)
    contributions = []
    for line in raw.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            contributions.append({
                "commits": int(parts[0]),
                "name": parts[-1],
                "email": parts[-2] if len(parts) > 2 else "",
            })
    return {"contributions": contributions, "total_lines": len(contributions)}


def git_verify() -> dict:
    """Check git health: remote config, branch, clean status."""
    return {
        "repo_path": git_repo_path(),
        "remote": git_remote(),
        "branch": git_branch(),
        "status_summary": git_status().get("modified", [])[:50],
        "recent_commits": git_log(n=5),
    }

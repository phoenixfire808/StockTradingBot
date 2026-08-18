"""AST-based code search, symbol extraction, and reference resolution."""

import ast
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _is_ignored(path: Path) -> bool:
    """Check if path matches .gitignore patterns."""
    ignore_file = PROJECT_ROOT / ".gitignore"
    if not ignore_file.exists():
        return False
    try:
        import fnmatch
        rel = str(path.relative_to(PROJECT_ROOT))
        lines = ignore_file.read_text().splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if fnmatch.fnmatch(rel, line):
                return True
            if fnmatch.fnmatch(path.name, line):
                return True
    except Exception:
        pass
    return False


def _iter_py_files(root: Path | None = None, recursive: bool = True) -> list[Path]:
    """Find all Python files respecting .gitignore."""
    root = root or PROJECT_ROOT
    files = []
    for p in root.rglob("*.py") if recursive else root.glob("*.py"):
        if not _is_ignored(p) and "/__pycache__/" not in str(p) and ".venv" not in str(p):
            files.append(p)
    return sorted(files)


def search_files(query: str, paths: str | None = None, exclude_patterns: str | None = None, max_results: int = 200) -> list[dict[str, Any]]:
    """Search file contents for text patterns with context lines."""
    results = []
    patterns_list = [p.strip() for p in query.split("|") if p.strip()]

    target_dirs = [Path(p.strip()) for p in paths.split(",") if p.strip()] if paths else [PROJECT_ROOT]

    excludes = [e.strip() for e in exclude_patterns.split(",") if e.strip()] if exclude_patterns else []

    for search_root in target_dirs:
        if not search_root.exists():
            continue
        for py_path in _iter_py_files(search_root, recursive=True):
            if any(str(exclude) in str(py_path) for exclude in excludes):
                continue
            try:
                content = py_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    for pat in patterns_list:
                        if re.search(pat, line, re.IGNORECASE):
                            start = max(0, i - 3)
                            end = min(len(lines), i + 4)
                            context_lines = [
                                (start + j, lines[start + j])
                                for j in range(end - start)
                            ]
                            results.append({
                                "file": str(py_path.relative_to(PROJECT_ROOT)),
                                "line": i + 1,
                                "match": line.strip(),
                                "context": context_lines,
                            })
                            break  # One match per line is enough
            except Exception:
                continue

    return results[:max_results]


def extract_symbols(file_path: str | None = None, content: str | None = None) -> list[dict]:
    """Extract Python symbols from a file using AST parsing."""
    if file_path:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    elif content:
        source = content
    else:
        raise ValueError("Provide file_path or content")

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [{"error": f"Syntax error: {e}"}]

    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            params = []
            for arg in node.args.args:
                if arg.arg != "self" and arg.arg != "cls":
                    params.append(arg.arg)
            decorators = [d.id if isinstance(d, ast.Name) else str(ast.dump(d)) for d in getattr(node, 'decorator_list', [])]
            symbols.append({
                "name": node.name,
                "kind": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", None),
                "params": params,
                "docstring": ast.get_docstring(node),
                "decorators": decorators,
            })
        elif isinstance(node, ast.ClassDef):
            bases = [b.id if isinstance(b, ast.Name) else str(ast.dump(b)) for b in getattr(node, 'bases', [])]
            symbols.append({
                "name": node.name,
                "kind": "class",
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", None),
                "bases": bases,
                "docstring": ast.get_docstring(node),
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbols.append({
                    "name": alias.asname or alias.name,
                    "kind": "import",
                    "line": node.lineno,
                    "source": alias.name,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                symbols.append({
                    "name": alias.asname or alias.name,
                    "kind": "import_from",
                    "line": node.lineno,
                    "source": f"{module}.{alias.name}" if module else alias.name,
                })
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append({
                        "name": target.id,
                        "kind": "variable",
                        "line": node.lineno,
                    })
    return symbols


def find_references(symbol_name: str, project_root: str | None = None, search_dir: str | None = None) -> list[dict]:
    """Find all references to a symbol across the codebase."""
    root = Path(project_root or str(PROJECT_ROOT))
    if search_dir:
        root = Path(search_dir)
    
    results = []
    for py_path in _iter_py_files(root):
        try:
            source = py_path.read_text(encoding="utf-8", errors="replace")
            lines = source.splitlines()
            for i, line in enumerate(lines):
                # Word boundary matching
                pattern = r'\b' + re.escape(symbol_name) + r'\b'
                if re.search(pattern, line):
                    results.append({
                        "file": str(py_path.relative_to(PROJECT_ROOT)),
                        "line": i + 1,
                        "content": line.strip(),
                    })
        except Exception:
            continue
    return results


def find_definitions(symbol_name: str) -> list[dict]:
    """Find where a symbol is defined by looking for def/class/assignment statements."""
    results = []
    for py_path in _iter_py_files():
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        
        for node in ast.walk(tree):
            defined = False
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
                defined = True
            elif isinstance(node, ast.ClassDef) and node.name == symbol_name:
                defined = True
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol_name:
                        defined = True
            
            if defined:
                results.append({
                    "file": str(py_path.relative_to(PROJECT_ROOT)),
                    "line": node.lineno,
                    "kind": type(node).__name__.replace("Def", "").lower(),
                })
    return results


def dependency_graph(module_path: str) -> dict:
    """Build import dependency graph for a module."""
    root = PROJECT_ROOT / module_path if not (PROJECT_ROOT / module_path).exists() else PROJECT_ROOT
    try:
        mod_file = next((f for f in _iter_py_files() if module_path in str(f)), None)
        if not mod_file:
            return {"error": f"Module path not found: {module_path}"}
        
        imports = []
        source = mod_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"type": "import", "module": alias.name, "as_name": alias.asname})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level or 0
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": f"{'.' * level}{module}" if level > 0 else module,
                        "imported": alias.name,
                        "as_name": alias.asname,
                    })
        
        return {"imports": imports, "direct_deps": len(imports)}
    except Exception as e:
        return {"error": str(e)}


def class_file_type(file_path: str) -> str:
    """Classify a file by its purpose based on name and content heuristics."""
    p = Path(file_path)
    name = p.name.lower()
    
    if "__init__" in name:
        return "package init"
    elif "test_" in name or "_test." in name:
        return "test file"
    elif name in ("server.py", "main.py", "app.py", "run.py"):
        return "entry point"
    elif "config" in name:
        return "configuration"
    elif "schema" in name:
        return "data schema"
    elif "utils" in str(p.parent).lower() or "helpers" in str(p.parent).lower():
        return "utility"
    elif "plugin" in name or "plugins" in str(p.parent).lower():
        return "plugin module"
    elif "mock" in name:
        return "mock/stub"
    elif "reference" in name or "docs" in str(p.parent).lower():
        return "documentation"
    else:
        return "source module"


def code_file_tree(depth: int = 3, show_hidden: bool = False) -> list[dict]:
    """Display project tree with depth limit and grouping."""
    root = PROJECT_ROOT
    result = []
    
    def _build_dir(dir_path: Path, current_depth: int) -> dict:
        entry = {
            "name": dir_path.name or "(root)",
            "type": "dir",
            "children": [],
        }
        if current_depth >= depth:
            return entry
        
        items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        excluded = {".git", ".venv", "__pycache__", "data", "reports", "logs", ".trash"}
        
        for item in items:
            if item.name.startswith(".") and not show_hidden:
                continue
            if item.name in excluded:
                continue
            
            if item.is_dir():
                child = _build_dir(item, current_depth + 1)
                entry["children"].append(child)
            else:
                ext = item.suffix.lstrip(".").lower() or "(none)"
                size = item.stat().st_size
                entry["children"].append({
                    "name": item.name,
                    "type": "file",
                    "extension": ext,
                    "size_bytes": size,
                })
        
        # Group children by extension for compact view
        grouped = {}
        for child in entry["children"]:
            if child["type"] == "dir":
                grouped.setdefault("__dirs__", []).append(child)
            else:
                ext = child.get("extension", "")
                grouped.setdefault(ext, []).append(child)
        
        # Reorder: dirs first, then files alphabetically
        ordered = []
        dirs = grouped.get("__dirs__", [])
        if dirs:
            ordered.extend(sorted(dirs, key=lambda d: d["name"]))
        
        all_files = []
        for ext, files in sorted(grouped.items()):
            if ext != "__dirs__":
                all_files.extend(sorted(files, key=lambda f: f["name"]))
        ordered.extend(all_files)
        entry["children"] = ordered
        
        return entry
    
    return [_build_dir(root, 0)]

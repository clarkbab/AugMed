#!/usr/bin/env python3
"""
Sort and clean import statements in Python source files.

Rules
-----
1. **External / third-party imports** come first, sorted alphabetically
   by the full import line.
2. A single blank line separates them from …
3. **Relative (augmed) imports**, ordered by descending dot-level
   (``....`` before ``...`` before ``..`` before ``.``), and
   alphabetically within each level.
4. Within each ``from X import a, B, c`` line, imported names are sorted
   alphabetically with **class names** (capitalised) before functions /
   variables (lower-case).
5. ``from X import *`` (star imports) are expanded to import only the
   names that are actually used in the file.
6. Unused imports are removed.

Usage
-----
    # Dry run — report issues without modifying files:
    python scripts/lint/sort_imports.py augmed/

    # Fix in-place:
    python scripts/lint/sort_imports.py augmed/ --fix

    # Fix a single file:
    python scripts/lint/sort_imports.py augmed/transforms/pipeline.py --fix

Exit codes
----------
    0 — all imports sorted (or --fix applied successfully)
    1 — unsorted / unused imports found (dry-run mode)
"""

import argparse
import ast
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

SKIP_PATTERNS = {'__pycache__', '.egg-info', 'node_modules', '.git', '.venv', 'venv'}


# ---------------------------------------------------------------------------
# Helpers — resolve star-importable names from a module
# ---------------------------------------------------------------------------

def _resolve_module_path(from_module: str, file_path: Path) -> Optional[Path]:
    """Resolve a relative import like ``...typing`` to the actual file."""
    # Count leading dots.
    dots = 0
    for ch in from_module:
        if ch == '.':
            dots += 1
        else:
            break
    remainder = from_module[dots:]  # e.g. "typing", "utils.args"

    # Walk up from the current file's package directory.
    # 1 dot = same package (no traversal), 2 dots = parent (1 up), etc.
    base = file_path.parent
    for _ in range(dots - 1):
        base = base.parent

    # Resolve the module within that base.
    parts = remainder.split('.') if remainder else []
    target = base.joinpath(*parts)

    # Could be a package (directory with __init__.py) or a module (.py).
    if target.is_dir() and (target / '__init__.py').exists():
        return target / '__init__.py'
    py = target.with_suffix('.py')
    if py.exists():
        return py
    return None


def _public_names_from_file(module_path: Path) -> Set[str]:
    """Return all public names defined/exported by *module_path*.

    Handles:
    - ``__all__`` if present.
    - Top-level assignments, function/class defs, and re-exports via
      ``from X import name`` or ``from X import *`` (one level only).
    """
    try:
        source = module_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, PermissionError):
        return set()

    # Check for __all__
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }

    names: Set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith('_'):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith('_'):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith('_'):
                names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            if node.names and node.names[0].name == '*':
                # Recursively resolve the star import (one level only).
                sub_path = _resolve_module_path(
                    '.' * (node.level or 0) + (node.module or ''),
                    module_path,
                )
                if sub_path:
                    names |= _public_names_from_file(sub_path)
            else:
                for alias in node.names:
                    n = alias.asname if alias.asname else alias.name
                    if not n.startswith('_'):
                        names.add(n)
        elif isinstance(node, ast.Import):
            # ``import X`` or ``import X as Y`` also contributes a public name.
            for alias in node.names:
                n = alias.asname if alias.asname else alias.name.split('.')[0]
                if not n.startswith('_'):
                    names.add(n)

    return names


def _bare_imports_from_file(module_path: Path) -> List[Tuple[str, Optional[str]]]:
    """Return ``(module, asname)`` pairs for top-level ``import X [as Y]``
    statements in *module_path*.  These are the transitive bare imports that
    a ``from module import *`` would make available."""
    try:
        source = module_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, PermissionError):
        return []

    results: List[Tuple[str, Optional[str]]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((alias.name, alias.asname))
    return results


def _public_names_from_module(module_name: str) -> Set[str]:
    """Return public names exported by an installed (stdlib / third-party) module."""
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return set()
    if hasattr(mod, '__all__'):
        return set(mod.__all__)
    return {n for n in dir(mod) if not n.startswith('_')}


def _expand_star_import(
    from_module: str,
    file_path: Path,
    used_names: Set[str],
) -> Tuple[Optional[List[str]], List['_Import']]:
    """Return (names_to_import, extra_bare_imports) to replace ``*``.

    *names_to_import* is the sorted list of ``from X import ...`` names,
    or ``None`` if the module could not be resolved.

    *extra_bare_imports* contains any ``import X as Y`` statements that
    the star-imported module makes available (e.g. ``import numpy as np``
    inside ``augmed/typing.py``).
    """
    extra: List[_Import] = []

    # Check if this is a relative import (has leading dots).
    dots = sum(1 for ch in from_module if ch == '.')
    if dots > 0:
        mod_path = _resolve_module_path(from_module, file_path)
        if mod_path is None:
            return None, []
        available = _public_names_from_file(mod_path)
        # Also collect bare ``import X [as Y]`` from the module.
        for mod, asname in _bare_imports_from_file(mod_path):
            check = asname or mod.split('.')[0]
            if check in used_names:
                extra.append(_Import(
                    alias=asname,
                    is_from=False,
                    level=0,
                    module=mod,
                    names=[(mod, asname)],
                    raw='',
                ))
    else:
        # Absolute import (stdlib / third-party).
        available = _public_names_from_module(from_module)
    if not available:
        return None, extra
    # Only keep names that are actually used.
    needed = sorted(available & used_names, key=_import_name_sort_key)
    return (needed if needed else None), extra


# ---------------------------------------------------------------------------
# Collect all names used in the file body (excluding import section)
# ---------------------------------------------------------------------------

def _collect_used_names(source: str, import_end_line: int) -> Set[str]:
    """Return all name tokens used in *source* after the import block."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: Set[str] = set()
    for node in ast.walk(tree):
        # Skip nodes that are part of the import block.
        if hasattr(node, 'lineno') and node.lineno <= import_end_line:
            # Still walk into the rest if the node spans beyond.
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue

        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For chained attributes like np.ndarray, collect 'np'.
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
        # Annotations in strings (forward references).
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Simple heuristic: if the string looks like a type annotation.
            for token in re.findall(r'[A-Za-z_]\w*', node.value):
                names.add(token)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Collect names from decorators and annotations.
            pass  # ast.walk handles them.

    return names


def _collect_all_names_in_source(source: str) -> Set[str]:
    """Return every Name node's id in the entire source (including type annotations)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for token in re.findall(r'[A-Za-z_]\w*', node.value):
                names.add(token)
    return names


# ---------------------------------------------------------------------------
# Parsing imports
# ---------------------------------------------------------------------------

_IMPORT_RE = re.compile(
    r'^(?:import\s|from\s)',
)


def _is_import_line(line: str) -> bool:
    """True if *line* starts an import statement."""
    stripped = line.lstrip()
    return bool(_IMPORT_RE.match(stripped))


def _find_import_block(lines: List[str]) -> Tuple[int, int]:
    """Return (start, end) 0-based line indices of the import block.

    The import block is the contiguous group of import lines (allowing
    blank lines between them) at the top of the file, after any leading
    docstrings / comments / ``from __future__`` imports.
    """
    first_import = None
    last_import = None

    i = 0
    n = len(lines)

    # Skip leading comments, blank lines, and docstrings.
    while i < n:
        stripped = lines[i].strip()
        if stripped == '' or stripped.startswith('#'):
            i += 1
            continue
        # Multi-line docstring at module level.
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and stripped.endswith(quote) and len(stripped) > 3:
                i += 1
                continue
            # Multi-line.
            i += 1
            while i < n and quote not in lines[i]:
                i += 1
            i += 1  # skip closing line
            continue
        break

    # Now gather imports.
    in_paren = False
    while i < n:
        stripped = lines[i].strip()

        if in_paren:
            last_import = i
            if ')' in stripped:
                in_paren = False
            i += 1
            continue

        if stripped == '' or stripped.startswith('#'):
            # Blank / comment lines inside the import block are OK.
            i += 1
            continue

        if _is_import_line(stripped):
            if first_import is None:
                first_import = i
            last_import = i
            if '(' in stripped and ')' not in stripped:
                in_paren = True
            i += 1
            continue

        # Non-import, non-blank — end of block.
        break

    if first_import is None:
        return (0, 0)
    return (first_import, last_import + 1)


# ---------------------------------------------------------------------------
# Import representation
# ---------------------------------------------------------------------------

class _Import:
    """Parsed representation of a single import statement."""
    __slots__ = ('is_from', 'module', 'level', 'names', 'alias', 'raw')

    def __init__(
        self,
        is_from: bool,
        module: Optional[str],
        level: int,
        names: List[Tuple[str, Optional[str]]],
        alias: Optional[str],
        raw: str,
    ) -> None:
        self.is_from = is_from
        self.module = module        # e.g. "typing", "..utils.args", None
        self.level = level          # 0 for absolute, 1+ for relative
        self.names = names          # [(name, asname), ...]
        self.alias = alias          # for ``import X as Y``
        self.raw = raw

    @property
    def full_module(self) -> str:
        """Dotted module string with leading dots."""
        dots = '.' * self.level
        return dots + (self.module or '')

    @property
    def is_relative(self) -> bool:
        return self.level > 0

    @property
    def is_star(self) -> bool:
        return len(self.names) == 1 and self.names[0][0] == '*'

    def __repr__(self) -> str:
        return f'_Import({self.raw!r})'


def _parse_imports(source: str, start: int, end: int, lines: List[str]) -> List[_Import]:
    """Parse import statements from lines[start:end] using AST."""
    # We parse the full source to get AST nodes, then filter by line range.
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imports: List[_Import] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.lineno - 1 < start or node.lineno - 1 >= end:
            continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                imp = _Import(
                    alias=alias.asname,
                    is_from=False,
                    level=0,
                    module=alias.name,
                    names=[(alias.name, alias.asname)],
                    raw=_reconstruct_import(node, lines),
                )
                imports.append(imp)
        else:
            imp = _Import(
                alias=None,
                is_from=True,
                level=node.level,
                module=node.module or '',
                names=[(a.name, a.asname) for a in node.names],
                raw=_reconstruct_import(node, lines),
            )
            imports.append(imp)

    return imports


def _reconstruct_import(node: ast.AST, lines: List[str]) -> str:
    """Get the raw source text for an import AST node."""
    start = node.lineno - 1
    end = node.end_lineno  # 1-based inclusive -> exclusive when slicing
    raw_lines = lines[start:end]
    return '\n'.join(line.rstrip() for line in raw_lines)


# ---------------------------------------------------------------------------
# Sort keys
# ---------------------------------------------------------------------------

def _import_name_sort_key(name: str) -> Tuple[int, str]:
    """Class names (upper-case start) before lower-case, then alphabetical."""
    # 0 for upper-case (classes), 1 for lower/underscore (functions, variables).
    return (0 if name[0].isupper() else 1, name.lower()) if name else (2, '')


def _sort_imported_names(names: List[Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
    """Sort a list of imported names: classes first, then alphabetical."""
    return sorted(names, key=lambda pair: _import_name_sort_key(pair[0]))


def _external_sort_key(imp: _Import) -> Tuple[int, str]:
    """Sort key for external (non-relative) imports.

    Sort alphabetically by module name. ``import X`` and ``from X import ...``
    are interleaved together by module name.
    """
    mod = (imp.module or '').lower()
    # ``from X`` (1) after ``import X`` (0) when same module.
    return (mod, 1 if imp.is_from else 0)


def _relative_sort_key(imp: _Import) -> Tuple[int, str]:
    """Sort key for relative imports: more dots first, then alphabetical."""
    # Negate level so higher dot-counts come first.
    return (-imp.level, (imp.module or '').lower())


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_import(imp: _Import) -> str:
    """Produce a canonical single-line import string."""
    if not imp.is_from:
        # ``import X`` or ``import X as Y``
        name, asname = imp.names[0]
        if asname:
            return f'import {name} as {asname}'
        return f'import {name}'

    # ``from X import ...``
    prefix = imp.full_module
    name_strs = []
    for name, asname in imp.names:
        if asname:
            name_strs.append(f'{name} as {asname}')
        else:
            name_strs.append(name)
    joined = ', '.join(name_strs)
    return f'from {prefix} import {joined}'


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def sort_imports(source: str, file_path: Path) -> Tuple[str, int]:
    """Sort and clean imports in *source*.  Returns (new_source, n_changes)."""
    lines = source.splitlines(keepends=True)
    start, end = _find_import_block(lines)

    if start == end:
        return source, 0

    parsed = _parse_imports(source, start, end, [l.rstrip('\n\r') for l in lines])
    if not parsed:
        return source, 0

    # Collect names used in the non-import body of the file.
    body_source_lines = lines[:start] + lines[end:]
    body_source = ''.join(body_source_lines)
    used_names = _collect_all_names_in_source(body_source)

    # Also add names used inside the import block itself that might reference
    # each other (e.g. type annotations that depend on imported names are in the body).

    # __init__.py files use star imports for re-exporting — don't touch them.
    is_init = file_path.name == '__init__.py'

    # --- Phase 0: Separate __future__ imports (always preserved) ---
    future_imports: List[_Import] = []
    non_future: List[_Import] = []
    for imp in parsed:
        if imp.is_from and not imp.is_relative and imp.module == '__future__':
            future_imports.append(imp)
        else:
            non_future.append(imp)
    parsed = non_future

    # --- Phase 1: Expand star imports ---
    expanded: List[_Import] = []
    for imp in parsed:
        if imp.is_star:
            if is_init:
                # Keep star imports in __init__.py (re-export pattern).
                expanded.append(imp)
                continue
            replacement, extra_imports = _expand_star_import(
                imp.full_module, file_path, used_names,
            )
            # Inject any bare ``import X [as Y]`` that the star module provided.
            expanded.extend(extra_imports)
            if replacement:
                imp = _Import(
                    alias=None,
                    is_from=True,
                    level=imp.level,
                    module=imp.module,
                    names=[(n, None) for n in replacement],
                    raw=imp.raw,
                )
            else:
                # Could not resolve — keep the star import as-is.
                continue
        expanded.append(imp)

    # --- Phase 2: Remove unused imports ---
    # Skip unused-import removal for __init__.py (re-export files).
    if is_init:
        cleaned = list(expanded)
    else:
        cleaned: List[_Import] = []
        for imp in expanded:
            if not imp.is_from:
                # ``import X as Y`` — check if Y (or X) is used.
                name = imp.alias or imp.names[0][0].split('.')[0]
                if name in used_names:
                    cleaned.append(imp)
                continue

            if imp.is_star:
                # Unresolvable star — keep as-is.
                cleaned.append(imp)
                continue

            # Filter out individual names that aren't used.
            kept: List[Tuple[str, Optional[str]]] = []
            for name, asname in imp.names:
                check = asname or name
                if check in used_names:
                    kept.append((name, asname))

            if kept:
                imp = _Import(
                    alias=None,
                    is_from=True,
                    level=imp.level,
                    module=imp.module,
                    names=kept,
                    raw=imp.raw,
                )
                cleaned.append(imp)

    # --- Phase 3: Deduplicate names across imports ---
    # If the same name is imported from multiple modules, keep it only in
    # the first import that provides it (external before relative, and
    # within relative higher dot-level first — matching the final order).
    # Pre-sort so dedup respects the intended output order.
    ext_temp: List[_Import] = []
    rel_temp: List[_Import] = []
    for imp in cleaned:
        if imp.is_relative:
            rel_temp.append(imp)
        else:
            ext_temp.append(imp)
    ext_temp.sort(key=_external_sort_key)
    rel_temp.sort(key=_relative_sort_key)
    ordered_for_dedup = ext_temp + rel_temp

    seen_names: Set[str] = set()
    deduped: List[_Import] = []
    for imp in ordered_for_dedup:
        if not imp.is_from or imp.is_star:
            name = imp.alias or imp.names[0][0].split('.')[0]
            seen_names.add(name)
            deduped.append(imp)
            continue
        kept: List[Tuple[str, Optional[str]]] = []
        for name, asname in imp.names:
            check = asname or name
            if check not in seen_names:
                seen_names.add(check)
                kept.append((name, asname))
        if kept:
            imp = _Import(
                alias=None,
                is_from=True,
                level=imp.level,
                module=imp.module,
                names=kept,
                raw=imp.raw,
            )
            deduped.append(imp)
    cleaned = deduped

    # --- Phase 3b: Add missing imports for used-but-not-imported names ---
    # After star expansion, some names used in the body may not be covered
    # by any import (e.g. the original file had an explicit partial list
    # instead of ``*``).  For each relative import already present, check
    # if its source module provides additional names that the body needs.
    if not is_init:
        already_imported: Set[str] = set()
        for imp in cleaned:
            if not imp.is_from or imp.is_star:
                name = imp.alias or imp.names[0][0].split('.')[0]
                already_imported.add(name)
            else:
                for name, asname in imp.names:
                    already_imported.add(asname or name)

        missing = used_names - already_imported
        if missing:
            # Build a map: module -> set of available names.
            module_available: Dict[str, Set[str]] = {}
            for imp in cleaned:
                if not imp.is_from or not imp.is_relative:
                    continue
                key = imp.full_module
                if key not in module_available:
                    mod_path = _resolve_module_path(key, file_path)
                    if mod_path:
                        module_available[key] = _public_names_from_file(mod_path)

            # For each missing name, find a module that provides it and
            # add it to that import.
            for key, available in module_available.items():
                to_add = missing & available
                if not to_add:
                    continue
                # Find the corresponding import and extend it.
                for i, imp in enumerate(cleaned):
                    if imp.is_from and imp.full_module == key:
                        new_names = list(imp.names) + [(n, None) for n in to_add]
                        cleaned[i] = _Import(
                            alias=None,
                            is_from=True,
                            level=imp.level,
                            module=imp.module,
                            names=new_names,
                            raw=imp.raw,
                        )
                        missing -= to_add
                        break

            # Also check stdlib / third-party modules for remaining missing names.
            if missing:
                for imp in cleaned:
                    if not imp.is_from or imp.is_relative:
                        continue
                    mod_name = imp.module or ''
                    if mod_name not in module_available:
                        module_available[mod_name] = _public_names_from_module(mod_name)
                    to_add = missing & module_available[mod_name]
                    if not to_add:
                        continue
                    new_names = list(imp.names) + [(n, None) for n in to_add]
                    idx = cleaned.index(imp)
                    cleaned[idx] = _Import(
                        alias=None,
                        is_from=True,
                        level=imp.level,
                        module=imp.module,
                        names=new_names,
                        raw=imp.raw,
                    )
                    missing -= to_add
    for imp in cleaned:
        if imp.is_from and not imp.is_star:
            imp.names = _sort_imported_names(imp.names)

    # --- Phase 5: Separate into external vs relative, and sort ---
    external: List[_Import] = []
    relative: List[_Import] = []
    for imp in cleaned:
        if imp.is_relative:
            relative.append(imp)
        else:
            external.append(imp)

    external.sort(key=_external_sort_key)
    relative.sort(key=_relative_sort_key)

    # --- Phase 6: Reconstruct the import block ---
    new_lines: List[str] = []
    for imp in future_imports:
        new_lines.append(_format_import(imp))
    if future_imports and (external or relative):
        new_lines.append('')
    for imp in external:
        new_lines.append(_format_import(imp))
    if external and relative:
        new_lines.append('')
    for imp in relative:
        new_lines.append(_format_import(imp))

    # Compare with original.
    original_import_text_lines = [l.rstrip('\n\r') for l in lines[start:end]]
    # Strip trailing blank lines from original import block.
    while original_import_text_lines and original_import_text_lines[-1].strip() == '':
        original_import_text_lines.pop()

    if new_lines == original_import_text_lines:
        return source, 0

    # Count changes for reporting.
    n_changes = 1  # At least an ordering change.

    # Rebuild the source.
    # Determine the newline style.
    nl = '\n'
    new_block = nl.join(new_lines) + nl
    prefix = ''.join(lines[:start])
    suffix = ''.join(lines[end:])

    new_source = prefix + new_block + suffix
    return new_source, n_changes


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def _should_skip(path: Path) -> bool:
    return any(
        any(skip in part for skip in SKIP_PATTERNS)
        for part in path.parts
    )


def process_file(path: Path, *, fix: bool = False) -> int:
    """Check / fix one file.  Returns count of issues."""
    try:
        source = path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return 0

    new_source, n = sort_imports(source, path.resolve())
    if n == 0:
        return 0

    if fix:
        path.write_text(new_source, encoding='utf-8')
        print(f'  Fixed imports in {path}')
    else:
        print(f'  Unsorted / unused imports in {path}')
    return n


def collect_files(paths: List[str], ext: str) -> List[Path]:
    """Expand directories and filter out generated / cache folders."""
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob(f'*{ext}')))
    return [f for f in files if not _should_skip(f)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Sort and clean import statements in Python source files.')
    parser.add_argument(
        'paths', default=['.'], help='Files or directories to process (default: cwd)',
        nargs='*')
    parser.add_argument(
        '--fix', action='store_true',
        help='Rewrite files in-place (default: report only)')
    parser.add_argument(
        '--ext', default='.py',
        help='File extension to scan (default: .py)')
    args = parser.parse_args()

    files = collect_files(args.paths, args.ext)
    mode = 'Fixing' if args.fix else 'Checking'
    print(f'{mode} {len(files)} file(s)...\n')

    total = sum(process_file(f, fix=args.fix) for f in files)

    print()
    if total == 0:
        print('All imports are sorted and clean.')
    elif args.fix:
        print(f'Fixed imports in {total} file(s) total.')
    else:
        print(f'Found issues in {total} file(s). Re-run with --fix to apply.')

    sys.exit(0 if (total == 0 or args.fix) else 1)


if __name__ == '__main__':
    main()

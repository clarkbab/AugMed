#!/usr/bin/env python3
"""
Sort function and method definitions alphabetically.

Rules
-----
- ``__init__`` always comes first.
- Methods/functions referenced in sibling decorator expressions come next,
  sorted alphabetically (e.g. ``ensure_loaded`` used as ``@ensure_loaded``,
  or ``to_numpy`` referenced in ``@bubble_args(to_numpy)``).
- Other methods/functions are sorted alphabetically, ignoring leading/trailing
  underscores (e.g. ``_foo`` sorts as ``foo``, ``__bar__`` as ``bar``).
- Consecutive same-name methods (e.g. ``@property`` getter/setter pairs)
  are kept together as a single sortable unit.
- Only contiguous runs of function/method definitions are sorted;
  assignments, inner classes, or other statements act as boundaries
  between sortable groups.
- Both class methods and module-level functions are sorted.

Usage
-----
    # Dry run — report unsorted methods without modifying files:
    python scripts/lint/sort_methods.py augmed/

    # Fix in-place:
    python scripts/lint/sort_methods.py augmed/ --fix

    # Fix a single file:
    python scripts/lint/sort_methods.py augmed/transforms/spatial/elastic.py --fix

Exit codes
----------
    0 — all methods sorted (or --fix applied successfully)
    1 — unsorted methods found (dry-run mode)

Notes
-----
- Decorators stay attached to their method.
- Separators (blank lines, comments) between methods keep their
  positional slots when methods are rearranged.
- The splice is length-preserving so multiple groups in the same file
  are safe to process in a single pass (bottom-up).
"""

import argparse
import ast
from pathlib import Path
import sys
from typing import List, Set, Tuple

SKIP_PATTERNS = {'__pycache__', '.egg-info', 'node_modules', '.git', '.venv', 'venv'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _method_start(node: ast.FunctionDef) -> int:
    """1-based line of the first decorator or ``def`` keyword."""
    if node.decorator_list:
        return min(d.lineno for d in node.decorator_list)
    return node.lineno


def _sort_key(name: str, decorator_names: Set[str] = frozenset()) -> Tuple[int, str]:
    """``__init__`` first, decorator methods second, then alphabetical ignoring underscores."""
    if name == '__init__':
        return (0, '')
    if name in decorator_names:
        return (1, name.strip('_').lower())
    return (2, name.strip('_').lower())


# ---------------------------------------------------------------------------
# Method unit — one or more consecutive same-name methods (property pairs)
# ---------------------------------------------------------------------------

class _MethodUnit:
    """Group of consecutive same-name methods treated as one sortable block."""
    __slots__ = ('name', 'nodes', 'start_line', 'end_line', '_decorator_names')

    def __init__(self, name: str, nodes: list, decorator_names: Set[str] = frozenset()) -> None:
        self.name = name
        self.nodes = nodes
        self.start_line = _method_start(nodes[0])
        self.end_line = nodes[-1].end_lineno
        self._decorator_names = decorator_names

    @property
    def sort_key(self) -> Tuple[int, str]:
        return _sort_key(self.name, self._decorator_names)


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _build_units(
    group: List[Tuple[int, ast.FunctionDef]],
    decorator_names: Set[str] = frozenset(),
) -> List[_MethodUnit]:
    """Merge consecutive same-name methods into sortable units."""
    units: List[_MethodUnit] = []
    for _idx, node in group:
        if units and units[-1].name == node.name:
            # Extend existing unit (e.g. property setter following getter).
            units[-1].nodes.append(node)
            units[-1].end_line = node.end_lineno
        else:
            units.append(_MethodUnit(node.name, [node], decorator_names))
    return units


def _collect_decorator_deps(body: list) -> Set[str]:
    """Return names of sibling functions/methods referenced in decorator expressions.

    This catches both direct usage (``@ensure_loaded``) and indirect
    references (``@bubble_args(to_numpy)``) — any sibling name that
    appears anywhere in a decorator AST subtree.
    """
    # Collect all names defined as functions in this body.
    defined = {
        item.name
        for item in body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # Walk every decorator AST subtree looking for Name references to siblings.
    deps: Set[str] = set()
    for item in body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in item.decorator_list:
            for node in ast.walk(dec):
                if isinstance(node, ast.Name) and node.id in defined and node.id != item.name:
                    deps.add(node.id)
    return deps


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
# Core detection
# ---------------------------------------------------------------------------

def _find_unsorted_groups(source: str) -> List[List[_MethodUnit]]:
    """Return groups of function/method units that are not alphabetically sorted."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: List[List[_MethodUnit]] = []

    # Module-level functions.
    _find_unsorted_in_body(tree.body, results)

    # Class methods.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _find_unsorted_in_body(node.body, results)

    # Process bottom-up so earlier line numbers stay valid.
    results.sort(key=lambda g: g[0].start_line, reverse=True)
    return results


def _find_unsorted_in_body(
    body: list,
    results: List[List[_MethodUnit]],
) -> None:
    """Append unsorted groups of functions/methods found in *body* to *results*."""
    methods = [
        (i, item)
        for i, item in enumerate(body)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(methods) < 2:
        return

    decorator_names = _collect_decorator_deps(body)

    for group in _group_consecutive(methods):
        units = _build_units(group, decorator_names)
        if len(units) < 2:
            continue
        keys = [u.sort_key for u in units]
        if keys != sorted(keys):
            results.append(units)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def _group_consecutive(
    methods: List[Tuple[int, ast.FunctionDef]],
) -> List[List[Tuple[int, ast.FunctionDef]]]:
    """Split *methods* into contiguous runs by their class-body index."""
    groups: List[List[Tuple[int, ast.FunctionDef]]] = []
    current = [methods[0]]
    for j in range(1, len(methods)):
        if methods[j][0] == methods[j - 1][0] + 1:
            current.append(methods[j])
        else:
            groups.append(current)
            current = [methods[j]]
    groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Sort method definitions in classes alphabetically.')
    parser.add_argument(
        'paths', default=['augmed', 'scripts'], help='Files or directories to process',
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
        print('All methods are alphabetically sorted.')
    elif args.fix:
        print(f'Fixed {total} group(s) total.')
    else:
        print(f'Found {total} unsorted group(s). Re-run with --fix to apply.')

    sys.exit(0 if (total == 0 or args.fix) else 1)


def process_file(path: Path, *, fix: bool = False) -> int:
    """Check / fix one file.  Returns count of unsorted groups."""
    try:
        source = path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return 0

    try:
        ast.parse(source)
    except SyntaxError as exc:
        print(f'  Syntax error in {path} (line {exc.lineno}): {exc.msg}')
        return 0

    new_source, n = sort_methods(source)
    if n == 0:
        return 0

    if fix:
        path.write_text(new_source, encoding='utf-8')
        print(f'  Fixed {n} group(s) in {path}')
    else:
        print(f'  Found {n} unsorted group(s) in {path}')
    return n


def _should_skip(path: Path) -> bool:
    return any(
        any(skip in part for skip in SKIP_PATTERNS)
        for part in path.parts
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def sort_methods(source: str) -> Tuple[str, int]:
    """Sort methods in *source*.  Returns ``(new_source, n_fixes)``."""
    groups = _find_unsorted_groups(source)
    if not groups:
        return source, 0

    lines = source.splitlines(keepends=True)

    for units in groups:
        # Extract each unit's text and the separators between units.
        unit_texts: List[List[str]] = []
        seps: List[List[str]] = []

        for i, unit in enumerate(units):
            unit_texts.append(lines[unit.start_line - 1 : unit.end_line])
            if i < len(units) - 1:
                sep_start = unit.end_line                       # 0-based index
                sep_end = units[i + 1].start_line - 1           # 0-based index (exclusive)
                seps.append(lines[sep_start : sep_end])

        # Determine sorted order.
        order = sorted(range(len(units)), key=lambda idx: units[idx].sort_key)
        sorted_texts = [unit_texts[i] for i in order]

        # Reassemble with original separators in positional slots.
        new_region: List[str] = []
        for i, text in enumerate(sorted_texts):
            new_region.extend(text)
            if i < len(seps):
                new_region.extend(seps[i])

        # Splice into source lines.
        region_start = units[0].start_line          # 1-based inclusive
        region_end = units[-1].end_line             # 1-based inclusive
        lines[region_start - 1 : region_end] = new_region

    return ''.join(lines), len(groups)


if __name__ == '__main__':
    main()

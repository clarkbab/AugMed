#!/usr/bin/env python3
"""
Sort keyword arguments in Python function calls alphabetically.

Scans Python source files for function calls with two or more keyword
arguments that are not in alphabetical order, and optionally rewrites
them in-place.

Usage
-----
    # Dry run — report unsorted calls without modifying files:
    python scripts/lint/sort_kwargs.py augmed/

    # Fix in-place:
    python scripts/lint/sort_kwargs.py augmed/ --fix

    # Fix a single file:
    python scripts/lint/sort_kwargs.py augmed/transforms/spatial/elastic.py --fix

Exit codes
----------
    0 — all kwargs sorted (or --fix applied successfully)
    1 — unsorted kwargs found (dry-run mode)

Notes
-----
- Only named keyword arguments are considered (``**kwargs`` is ignored).
- The script preserves all formatting (indentation, separators, trailing
  commas) because it rearranges the original text fragments rather than
  regenerating them.
- Each replacement is length-preserving (the same kwarg texts and
  separators are reused in a different order), so edits never invalidate
  earlier character offsets.
"""

import argparse
import ast
from pathlib import Path
import sys
from typing import List, Tuple

SKIP_PATTERNS = {'__pycache__', '.egg-info', 'node_modules', '.git', '.venv', 'venv'}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _build_line_offsets(source: str) -> List[int]:
    """Map each 0-based line index to its starting character offset."""
    offsets = [0]
    for i, ch in enumerate(source):
        if ch == '\n':
            offsets.append(i + 1)
    return offsets


def _char_offset(line_offsets: List[int], lineno: int, col: int) -> int:
    """Convert a 1-based line number and 0-based column to a char offset."""
    return line_offsets[lineno - 1] + col


def _find_unsorted_calls(source: str) -> List[Tuple[ast.Call, List[ast.keyword]]]:
    """Return ``(Call, [keyword, ...])`` pairs whose named kwargs are not sorted."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        named = [kw for kw in node.keywords if kw.arg is not None]
        if len(named) < 2:
            continue
        if [kw.arg for kw in named] != sorted(kw.arg for kw in named):
            results.append((node, named))

    # Sort by source position (earliest first); we reverse later for safe editing.
    results.sort(key=lambda pair: (pair[1][0].lineno, pair[1][0].col_offset))
    return results


def sort_kwargs(source: str) -> Tuple[str, int]:
    """Sort kwargs in *source*.  Returns ``(new_source, n_fixes)``."""
    unsorted = _find_unsorted_calls(source)
    if not unsorted:
        return source, 0

    offsets = _build_line_offsets(source)
    n_fixed = 0

    # Process end-to-start so earlier char offsets stay valid.
    for _call_node, kwargs in reversed(unsorted):
        sorted_names = sorted(kw.arg for kw in kwargs)

        # Extract the ``name=value`` text for each kwarg.
        kw_text = {}
        for kw in kwargs:
            s = _char_offset(offsets, kw.lineno, kw.col_offset)
            e = _char_offset(offsets, kw.end_lineno, kw.end_col_offset)
            kw_text[kw.arg] = source[s:e]

        # Extract separators (commas, whitespace, newlines) between kwargs.
        seps: List[str] = []
        for i in range(len(kwargs) - 1):
            s = _char_offset(offsets, kwargs[i].end_lineno, kwargs[i].end_col_offset)
            e = _char_offset(offsets, kwargs[i + 1].lineno, kwargs[i + 1].col_offset)
            seps.append(source[s:e])

        # Build the replacement with sorted kwargs and original separators.
        parts = [kw_text[sorted_names[0]]]
        for i in range(1, len(sorted_names)):
            parts.append(seps[i - 1])
            parts.append(kw_text[sorted_names[i]])
        new_region = ''.join(parts)

        # Splice into source.
        region_start = _char_offset(offsets, kwargs[0].lineno, kwargs[0].col_offset)
        region_end = _char_offset(offsets, kwargs[-1].end_lineno, kwargs[-1].end_col_offset)
        source = source[:region_start] + new_region + source[region_end:]
        n_fixed += 1

    return source, n_fixed


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

def _should_skip(path: Path) -> bool:
    return any(
        any(skip in part for skip in SKIP_PATTERNS)
        for part in path.parts
    )


def process_file(path: Path, *, fix: bool = False) -> int:
    """Check / fix one file.  Returns count of unsorted calls."""
    try:
        source = path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return 0

    new_source, n = sort_kwargs(source)
    if n == 0:
        return 0

    if fix:
        path.write_text(new_source, encoding='utf-8')
        print(f'  Fixed {n} call(s) in {path}')
    else:
        print(f'  Found {n} unsorted call(s) in {path}')
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
        description='Sort keyword arguments in function calls alphabetically.')
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
        print('All kwargs are alphabetically sorted.')
    elif args.fix:
        print(f'Fixed {total} call(s) total.')
    else:
        print(f'Found {total} unsorted call(s). Re-run with --fix to apply.')

    sys.exit(0 if (total == 0 or args.fix) else 1)


if __name__ == '__main__':
    main()

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
import builtins
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


def _defined_names_in_file(module_path: Path) -> Set[str]:
    """Return names that are *directly defined* in *module_path*.

    Only considers ``class``, ``def``, and top-level assignments — NOT
    re-exports via ``from X import Y``.
    """
    try:
        source = module_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, PermissionError):
        return set()

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
    return names


def _class_names_in_file(module_path: Path) -> Set[str]:
    """Return names of classes directly defined in *module_path*."""
    try:
        source = module_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, PermissionError):
        return set()

    return {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.ClassDef) and not node.name.startswith('_')
    }


def _collect_relative_imports_from_file(module_path: Path) -> Set[Path]:
    """Return the set of resolved file paths that *module_path* imports from
    via relative imports (excluding ``TYPE_CHECKING`` blocks).

    When a relative import resolves to an ``__init__.py`` that is an
    *ancestor package* of *module_path* (i.e. the importing file lives
    inside that package), the ``__init__.py`` is excluded because Python
    guarantees it is already loaded.  Non-ancestor ``__init__.py`` files
    (e.g. sibling or child packages) are included normally.
    """
    try:
        source = module_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, PermissionError):
        return set()

    # Identify line numbers inside ``if TYPE_CHECKING:`` blocks so we can
    # skip them — those imports don't execute at runtime and therefore
    # cannot contribute to a circular-import chain.
    tc_lines: Set[int] = set()
    for node in ast.iter_child_nodes(tree):
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == 'TYPE_CHECKING'):
            for child in node.body:
                for ln in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                    tc_lines.add(ln)

    # Compute the set of ancestor __init__.py files for module_path.
    own_ancestors = _ancestor_init_files(module_path)

    paths: Set[Path] = set()
    for node in ast.iter_child_nodes(tree):
        # Skip ``if TYPE_CHECKING:`` blocks (type-only, not runtime).
        if isinstance(node, ast.If):
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.lineno in tc_lines:
            continue
        if (node.level or 0) == 0:
            continue  # absolute import
        mod_str = '.' * node.level + (node.module or '')
        resolved = _resolve_module_path(mod_str, module_path)
        if resolved is not None:
            real = resolved.resolve()
            # Skip ancestor __init__.py — already loaded when module_path
            # is being imported.
            if real in own_ancestors:
                continue
            paths.add(real)
    return paths


def _ancestor_init_files(file_path: Path) -> Set[Path]:
    """Return the set of ``__init__.py`` files in ancestor directories of
    *file_path* (i.e. the package hierarchy above the file).

    These are guaranteed to be already loaded by the time *file_path* is
    imported, so they don't contribute to circular-import chains."""
    result: Set[Path] = set()
    d = file_path.resolve().parent
    while True:
        init = d / '__init__.py'
        if init.exists():
            result.add(init.resolve())
            d = d.parent
        else:
            break
    return result


def _would_cause_circular_import(file_path: Path, target_module: str) -> bool:
    """Return True if importing *target_module* (a relative import string
    like ``....typing``) from *file_path* would create a circular import.

    This performs a breadth-first walk of the runtime relative-import graph
    starting from the resolved *target_module*, checking whether any
    transitive import leads back to *file_path*.
    """
    resolved = _resolve_module_path(target_module, file_path)
    if resolved is None:
        return False  # can't resolve — assume safe

    target_resolved = resolved.resolve()
    origin = file_path.resolve()

    if target_resolved == origin:
        return True  # direct self-import

    # BFS with a depth limit to avoid very expensive traversals.
    visited: Set[Path] = {target_resolved}
    frontier: Set[Path] = {target_resolved}
    max_depth = 10
    for _ in range(max_depth):
        next_frontier: Set[Path] = set()
        for mod_path in frontier:
            for dep in _collect_relative_imports_from_file(mod_path):
                if dep == origin:
                    return True
                if dep not in visited:
                    visited.add(dep)
                    next_frontier.add(dep)
        if not next_frontier:
            break
        frontier = next_frontier

    return False


def _find_defining_module(
    name: str,
    file_path: Path,
) -> Optional[Tuple[int, str]]:
    """Search the project package for the file that *defines* *name*.

    Returns ``(level, module_dotted)`` suitable for building a relative
    ``from`` import, or ``None`` if not found.

    The search starts from the current file's package and walks up to the
    project root (the highest directory that still has ``__init__.py``).
    """
    # Find the project root (top-level package).
    pkg_root = file_path.parent
    while (pkg_root.parent / '__init__.py').exists():
        pkg_root = pkg_root.parent

    # Recursively scan all .py files under the package root.
    candidates: List[Tuple[Path, int]] = []
    for py_file in sorted(pkg_root.rglob('*.py')):
        if py_file == file_path:
            continue
        if any(skip in py_file.parts for skip in SKIP_PATTERNS):
            continue
        # Skip __init__.py files — names there are typically re-exports.
        if py_file.name == '__init__.py':
            continue
        if name in _defined_names_in_file(py_file):
            candidates.append(py_file)

    if not candidates:
        return None

    # Prefer the shortest module path (closest definition).
    best = min(candidates, key=lambda p: len(p.parts))

    # Compute the relative import from file_path to best.
    return _relative_import_for(file_path, best)


def _relative_import_for(
    from_file: Path,
    to_file: Path,
) -> Tuple[int, str]:
    """Compute ``(level, module_dotted)`` for a relative import.

    Given that we are in *from_file* and want to import from *to_file*,
    return the dot-level and module portion (e.g. ``(3, 'patient')`` for
    ``from ...patient import ...``).
    """
    from_parts = list(from_file.parent.parts)
    # If to_file is an __init__.py, the module is the directory itself.
    if to_file.name == '__init__.py':
        to_parts = list(to_file.parent.parts)
        to_module_parts: List[str] = []
    else:
        to_parts = list(to_file.parent.parts)
        to_module_parts = [to_file.stem]

    # Find common prefix length.
    common = 0
    for a, b in zip(from_parts, to_parts):
        if a == b:
            common += 1
        else:
            break

    # Level = how many directories we must go UP from from_file's package.
    # In Python's relative import system, 1 dot = current package (no
    # upward traversal), 2 dots = parent, etc.
    level = len(from_parts) - common + 1
    # Module = the remaining path components of to_file's directory + stem.
    remaining_dirs = to_parts[common:]
    module_parts = remaining_dirs + to_module_parts
    module_dotted = '.'.join(module_parts)

    return (level, module_dotted)


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

def _collect_annotation_string_ids(tree: ast.AST) -> Set[int]:
    """Return the ``id()`` of every ``ast.Constant`` string node that appears
    in a type-annotation context (forward references).

    This covers function parameter annotations, return annotations, variable
    annotations, and base classes — but NOT regular strings in function bodies,
    f-strings, docstrings, error messages, etc.
    """
    ids: Set[int] = set()

    def _walk_annotation(node: ast.AST) -> None:
        """Mark all Constant-string nodes inside an annotation subtree."""
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                ids.add(id(child))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns:
                _walk_annotation(node.returns)
            for arg in (node.args.args + node.args.posonlyargs +
                        node.args.kwonlyargs):
                if arg.annotation:
                    _walk_annotation(arg.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                _walk_annotation(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                _walk_annotation(node.args.kwarg.annotation)
        elif isinstance(node, ast.AnnAssign) and node.annotation:
            _walk_annotation(node.annotation)

    return ids


def _collect_used_names(source: str, import_end_line: int) -> Set[str]:
    """Return all name tokens used in *source* after the import block."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    annotation_str_ids = _collect_annotation_string_ids(tree)

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
        # Only extract tokens from strings in annotation positions.
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in annotation_str_ids:
                for token in re.findall(r'[A-Za-z_]\w*', node.value):
                    names.add(token)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Collect names from decorators and annotations.
            pass  # ast.walk handles them.

    return names


def _collect_all_names_in_source(source: str) -> Set[str]:
    """Return every Name node's id in the entire source (including type annotations).

    For string literals, only tokens inside annotation contexts (forward
    references) are included — regular strings such as error messages and
    parameter values are ignored.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    annotation_str_ids = _collect_annotation_string_ids(tree)

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
            if id(node) in annotation_str_ids:
                for token in re.findall(r'[A-Za-z_]\w*', node.value):
                    names.add(token)
    return names


def _collect_ast_name_references(source: str) -> Set[str]:
    """Return only names that appear as real ``ast.Name`` or ``ast.Attribute``
    root references in *source* — excludes tokens extracted from string literals."""
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
    return names


def _names_from_target(target: ast.AST) -> Set[str]:
    """Recursively extract all ``ast.Name`` ids from an assignment target
    (handles nested tuples/lists like ``for i, (m, u) in ...``)."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        result: Set[str] = set()
        for elt in target.elts:
            result |= _names_from_target(elt)
        return result
    if isinstance(target, ast.Starred):
        return _names_from_target(target.value)
    return set()


def _collect_defined_names(source: str) -> Set[str]:
    """Return names defined locally in *source* (functions, classes, assignments,
    params, exception handlers, context managers, and body-level imports such as
    those inside ``if TYPE_CHECKING`` blocks)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            # Add parameter names.
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                names.add(arg.arg)
            if node.args.vararg:
                names.add(node.args.vararg.arg)
            if node.args.kwarg:
                names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                names.add(arg.arg)
            if node.args.vararg:
                names.add(node.args.vararg.arg)
            if node.args.kwarg:
                names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= _names_from_target(target)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names |= _names_from_target(node.target)
        elif isinstance(node, ast.comprehension):
            names |= _names_from_target(node.target)
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        # ``except SomeError as e:``
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                names.add(node.name)
        # ``with open(f) as fh:``
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names |= _names_from_target(item.optional_vars)
        # Body-level imports (e.g. inside ``if TYPE_CHECKING:`` blocks).
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)

    # Common dunder attributes that are always available at module level.
    names.update({'__file__', '__name__', '__doc__', '__package__',
                  '__spec__', '__loader__', '__path__'})
    return names


_BUILTIN_NAMES: Set[str] = set(dir(builtins))


def _has_future_annotations(source: str) -> bool:
    """Return True if the source contains ``from __future__ import annotations``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.iter_child_nodes(tree):
        if (isinstance(node, ast.ImportFrom)
                and node.module == '__future__'
                and any(alias.name == 'annotations' for alias in node.names)):
            return True
    return False


def _collect_annotation_only_names(source: str) -> Set[str]:
    """Return names that appear ONLY in type-annotation contexts.

    A name is "annotation-only" when it is referenced in function parameter
    annotations, return-type annotations, or variable annotations but
    **never** in runtime expressions (calls, assignments, attribute access,
    etc.).

    This function works on the *body* source (import block already stripped).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    annotation_names: Set[str] = set()
    runtime_names: Set[str] = set()

    def _extract_names(node: ast.AST) -> Set[str]:
        """Extract all Name references from an AST subtree."""
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                root = child
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    names.add(root.id)
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                for token in re.findall(r'[A-Za-z_]\w*', child.value):
                    names.add(token)
        return names

    def _walk_runtime(node: ast.AST) -> None:
        """Walk a node that is in a runtime context, collecting names."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                runtime_names.add(child.id)
            elif isinstance(child, ast.Attribute):
                root = child
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    runtime_names.add(root.id)
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                for token in re.findall(r'[A-Za-z_]\w*', child.value):
                    runtime_names.add(token)

    for node in ast.walk(tree):
        # Function / method definitions: annotations are type-only, body is runtime.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Return annotation.
            if node.returns:
                annotation_names |= _extract_names(node.returns)
            # Argument annotations.
            for arg in (node.args.args + node.args.posonlyargs +
                        node.args.kwonlyargs):
                if arg.annotation:
                    annotation_names |= _extract_names(arg.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                annotation_names |= _extract_names(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                annotation_names |= _extract_names(node.args.kwarg.annotation)
            # Default values are runtime.
            for default in node.args.defaults + node.args.kw_defaults:
                if default:
                    _walk_runtime(default)
            # Decorators are runtime.
            for decorator in node.decorator_list:
                _walk_runtime(decorator)
            # Body statements are handled by further walk iterations.

        elif isinstance(node, ast.AnnAssign):
            # ``x: SomeType = value`` — the annotation is type-only,
            # the value (if any) is runtime.
            if node.annotation:
                annotation_names |= _extract_names(node.annotation)
            if node.value:
                _walk_runtime(node.value)

        elif isinstance(node, ast.ClassDef):
            # Base classes are RUNTIME (evaluated when the class is created).
            for base in node.bases:
                _walk_runtime(base)
            for kw in node.keywords:
                _walk_runtime(kw.value)
            for decorator in node.decorator_list:
                _walk_runtime(decorator)

    # Now collect ALL names from the source as runtime, then subtract
    # annotation positions. The simplest approach: collect runtime names
    # from every context that is NOT an annotation.
    # We already collected annotation_names above. Now we need all the
    # runtime references. We do a second pass where we walk only
    # non-annotation subtrees.
    for node in ast.iter_child_nodes(tree):
        # Skip import nodes.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        # For function defs, walk body + decorators + defaults but NOT annotations.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                _walk_runtime(decorator)
            for default in node.args.defaults + node.args.kw_defaults:
                if default:
                    _walk_runtime(default)
            for child in node.body:
                _walk_runtime(child)
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                _walk_runtime(base)
            for kw in node.keywords:
                _walk_runtime(kw.value)
            for decorator in node.decorator_list:
                _walk_runtime(decorator)
            for child in node.body:
                # Class body items — recurse to handle methods etc.
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in child.decorator_list:
                        _walk_runtime(dec)
                    for default in child.args.defaults + child.args.kw_defaults:
                        if default:
                            _walk_runtime(default)
                    for sub in child.body:
                        _walk_runtime(sub)
                elif isinstance(child, ast.AnnAssign):
                    if child.value:
                        _walk_runtime(child.value)
                else:
                    _walk_runtime(child)
        elif isinstance(node, ast.AnnAssign):
            if node.value:
                _walk_runtime(node.value)
        else:
            _walk_runtime(node)

    # Annotation-only = names that appear in annotations but NOT in runtime.
    return annotation_names - runtime_names


def _collect_string_annotation_names(source: str) -> Set[str]:
    """Return class-like names that appear as *string* annotations.

    These are annotations written as ``param: 'SomeClass'`` — typically
    forward references used to avoid circular imports.  Only names whose
    first character is upper-case are returned (class names).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: Set[str] = set()

    def _scan_annotation(node: ast.AST) -> None:
        """If *node* is a string constant, extract class-like tokens."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for token in re.findall(r'[A-Za-z_]\w*', node.value):
                if token[0].isupper():
                    names.add(token)
        # Handle ``'A' | 'B'`` (BinOp with | in 3.10+) etc.
        elif isinstance(node, ast.BinOp):
            _scan_annotation(node.left)
            _scan_annotation(node.right)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns:
                _scan_annotation(node.returns)
            for arg in (node.args.args + node.args.posonlyargs +
                        node.args.kwonlyargs):
                if arg.annotation:
                    _scan_annotation(arg.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                _scan_annotation(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                _scan_annotation(node.args.kwarg.annotation)
        elif isinstance(node, ast.AnnAssign) and node.annotation:
            _scan_annotation(node.annotation)

    return names


def _unquote_annotations(source: str) -> str:
    """Replace string-literal annotations with bare names.

    After ``from __future__ import annotations`` is added, annotations are
    no longer evaluated at runtime, so forward references written as
    ``'SomeClass'`` can become ``SomeClass``.

    Only single-quoted or double-quoted annotations that consist of a valid
    Python expression are unquoted.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Collect (start_offset, end_offset, replacement) for each string
    # annotation that should be unquoted.  Work on byte offsets in the
    # source text.
    lines = source.splitlines(keepends=True)
    replacements: List[Tuple[int, int, str]] = []

    def _line_col_to_offset(lineno: int, col: int) -> int:
        return sum(len(lines[i]) for i in range(lineno - 1)) + col

    def _maybe_unquote(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Verify the string looks like a valid type expression.
            try:
                ast.parse(node.value, mode='eval')
            except SyntaxError:
                return
            start = _line_col_to_offset(node.lineno, node.col_offset)
            end = _line_col_to_offset(node.end_lineno, node.end_col_offset)
            replacements.append((start, end, node.value))
        elif isinstance(node, ast.BinOp):
            _maybe_unquote(node.left)
            _maybe_unquote(node.right)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns:
                _maybe_unquote(node.returns)
            for arg in (node.args.args + node.args.posonlyargs +
                        node.args.kwonlyargs):
                if arg.annotation:
                    _maybe_unquote(arg.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                _maybe_unquote(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                _maybe_unquote(node.args.kwarg.annotation)
        elif isinstance(node, ast.AnnAssign) and node.annotation:
            _maybe_unquote(node.annotation)

    if not replacements:
        return source

    # Apply replacements in reverse order to preserve offsets.
    replacements.sort(key=lambda r: r[0], reverse=True)
    chars = list(source)
    for start, end, replacement in replacements:
        chars[start:end] = list(replacement)

    return ''.join(chars)


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

        # ``if TYPE_CHECKING:`` guards are part of the import block.
        if stripped == 'if TYPE_CHECKING:':
            if first_import is None:
                first_import = i
            last_import = i
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
    __slots__ = ('is_from', 'module', 'level', 'names', 'alias', 'raw',
                 'type_checking')

    def __init__(
        self,
        is_from: bool,
        module: Optional[str],
        level: int,
        names: List[Tuple[str, Optional[str]]],
        alias: Optional[str],
        raw: str,
        type_checking: bool = False,
    ) -> None:
        self.is_from = is_from
        self.module = module        # e.g. "typing", "..utils.args", None
        self.level = level          # 0 for absolute, 1+ for relative
        self.names = names          # [(name, asname), ...]
        self.alias = alias          # for ``import X as Y``
        self.raw = raw
        self.type_checking = type_checking  # True if inside ``if TYPE_CHECKING:``

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

    # Build a set of line numbers that are inside ``if TYPE_CHECKING:`` blocks.
    tc_import_lines: Set[int] = set()
    for node in ast.iter_child_nodes(tree):
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == 'TYPE_CHECKING'
                and node.lineno - 1 >= start
                and node.lineno - 1 < end):
            for child in node.body:
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    for ln in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                        tc_import_lines.add(ln)

    imports: List[_Import] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            # Also handle ``if TYPE_CHECKING:`` nodes — parse their children.
            if (isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name)
                    and node.test.id == 'TYPE_CHECKING'):
                for child in node.body:
                    if not isinstance(child, (ast.Import, ast.ImportFrom)):
                        continue
                    if child.lineno - 1 < start or child.lineno - 1 >= end:
                        continue
                    if isinstance(child, ast.Import):
                        for alias in child.names:
                            imp = _Import(
                                alias=alias.asname,
                                is_from=False,
                                level=0,
                                module=alias.name,
                                names=[(alias.name, alias.asname)],
                                raw=_reconstruct_import(child, lines),
                                type_checking=True,
                            )
                            imports.append(imp)
                    else:
                        imp = _Import(
                            alias=None,
                            is_from=True,
                            level=child.level,
                            module=child.module or '',
                            names=[(a.name, a.asname) for a in child.names],
                            raw=_reconstruct_import(child, lines),
                            type_checking=True,
                        )
                        imports.append(imp)
            continue
        if node.lineno - 1 < start or node.lineno - 1 >= end:
            continue
        # Skip imports that are inside TYPE_CHECKING blocks — already handled.
        if node.lineno in tc_import_lines:
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
                    type_checking=imp.type_checking,
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
            # TYPE_CHECKING imports are only for annotations — always keep.
            if imp.type_checking:
                cleaned.append(imp)
                continue

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
                    type_checking=imp.type_checking,
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
                type_checking=imp.type_checking,
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

        # Use only real AST name references (not tokens from string literals)
        # and exclude locally-defined names and builtins to avoid adding
        # spurious imports.
        ast_refs = _collect_ast_name_references(body_source)
        defined = _collect_defined_names(body_source)
        missing = ast_refs - already_imported - _BUILTIN_NAMES - defined
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

            # Warn about remaining names that are genuine AST references
            # but not found in any imported module.
            if missing:
                for name in sorted(missing):
                    print(f'  Warning: {file_path}: used name {name!r} not found in any imported module')

    for imp in cleaned:
        if imp.is_from and not imp.is_star:
            imp.names = _sort_imported_names(imp.names)

    # --- Phase 4b: Detect type-checking-only imports ---
    # Annotation-only names from *relative* imports that would cause a
    # circular import are moved into an ``if TYPE_CHECKING:`` block.
    # We only guard imports whose target module transitively imports back
    # into *file_path* (i.e. would create a real circular dependency).
    # Imports from modules that don't form a cycle (e.g. type aliases in
    # a ``typing`` module, or classes in unrelated modules) are left as
    # normal runtime imports even if only used in annotations.
    #
    # When imports are moved to TYPE_CHECKING, ``from __future__ import
    # annotations`` is also added (if absent) so that annotations become
    # strings and don't trigger an import at runtime.
    if not is_init:
        annotation_only = _collect_annotation_only_names(body_source)
        if annotation_only:
            # Cache circular-import checks per module to avoid repeated BFS.
            _circular_cache: Dict[str, bool] = {}
            new_cleaned: List[_Import] = []
            for imp in cleaned:
                if imp.type_checking:
                    # Already in TYPE_CHECKING — keep as-is.
                    new_cleaned.append(imp)
                    continue

                # Only consider relative (project-internal) imports.
                if not imp.is_relative:
                    new_cleaned.append(imp)
                    continue

                # Check whether this import's module would cause a
                # circular import.  Only guard it if so.
                mod_key = imp.full_module
                if mod_key not in _circular_cache:
                    _circular_cache[mod_key] = _would_cause_circular_import(
                        file_path, mod_key)
                if not _circular_cache[mod_key]:
                    new_cleaned.append(imp)
                    continue

                if not imp.is_from or imp.is_star:
                    # Bare ``import X`` — check if the name is
                    # used only in annotations.
                    name = imp.alias or imp.names[0][0].split('.')[0]
                    if name in annotation_only:
                        imp = _Import(
                            alias=imp.alias,
                            is_from=imp.is_from,
                            level=imp.level,
                            module=imp.module,
                            names=imp.names,
                            raw=imp.raw,
                            type_checking=True,
                        )
                    new_cleaned.append(imp)
                    continue

                # ``from .X import A, b, C`` — split into runtime vs
                # type-only.
                runtime_names: List[Tuple[str, Optional[str]]] = []
                tc_names: List[Tuple[str, Optional[str]]] = []
                for name, asname in imp.names:
                    check = asname or name
                    if check in annotation_only:
                        tc_names.append((name, asname))
                    else:
                        runtime_names.append((name, asname))

                if runtime_names:
                    new_cleaned.append(_Import(
                        alias=None,
                        is_from=True,
                        level=imp.level,
                        module=imp.module,
                        names=runtime_names,
                        raw=imp.raw,
                    ))
                if tc_names:
                    new_cleaned.append(_Import(
                        alias=None,
                        is_from=True,
                        level=imp.level,
                        module=imp.module,
                        names=tc_names,
                        raw=imp.raw,
                        type_checking=True,
                    ))
            cleaned = new_cleaned

    # --- Phase 4c: Import string-annotated class names under TYPE_CHECKING ---
    # Detect class names that appear as string annotations (e.g.
    # ``param: 'DicomPatient'``) but are not currently imported.  Resolve
    # the *defining* module (where the class/function is actually defined,
    # not just re-exported) and add an import.  If the import would cause
    # a circular dependency, place it under ``if TYPE_CHECKING:``;
    # otherwise add it as a normal runtime import.
    if not is_init:
        str_ann_names = _collect_string_annotation_names(body_source)
        if str_ann_names:
            already_imported: Set[str] = set()
            for imp in cleaned:
                if not imp.is_from or imp.is_star:
                    name = imp.alias or imp.names[0][0].split('.')[0]
                    already_imported.add(name)
                else:
                    for name, asname in imp.names:
                        already_imported.add(asname or name)
            unresolved = str_ann_names - already_imported - _BUILTIN_NAMES
            if unresolved:
                # Group names by their defining module.
                # module_key (level, module_dotted) -> list of names
                module_names: Dict[Tuple[int, str], List[str]] = {}
                still_unresolved: Set[str] = set()
                for uname in unresolved:
                    result = _find_defining_module(uname, file_path)
                    if result is not None:
                        module_names.setdefault(result, []).append(uname)
                    else:
                        still_unresolved.add(uname)

                for (level, mod_dotted), names_list in module_names.items():
                    full_key = '.' * level + mod_dotted
                    # Determine if this import would cause a circular dependency.
                    is_circular = _would_cause_circular_import(file_path, full_key)
                    # Check if there's already a matching import for this module.
                    found = False
                    for i, imp in enumerate(cleaned):
                        if (imp.is_from
                                and imp.type_checking == is_circular
                                and imp.full_module == full_key):
                            new_names = list(imp.names) + [(n, None) for n in names_list]
                            cleaned[i] = _Import(
                                alias=None,
                                is_from=True,
                                level=imp.level,
                                module=imp.module,
                                names=_sort_imported_names(new_names),
                                raw=imp.raw,
                                type_checking=is_circular,
                            )
                            found = True
                            break
                    if not found:
                        cleaned.append(_Import(
                            alias=None,
                            is_from=True,
                            level=level,
                            module=mod_dotted,
                            names=_sort_imported_names(
                                [(n, None) for n in names_list]),
                            raw='',
                            type_checking=is_circular,
                        ))

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

    # If we have TYPE_CHECKING imports, ensure ``TYPE_CHECKING`` is imported
    # from ``typing`` and ``from __future__ import annotations`` is present
    # (so that annotations are not evaluated at runtime).
    has_tc_imports = any(imp.type_checking for imp in cleaned)
    if has_tc_imports:
        # Ensure ``from __future__ import annotations``.
        has_future_annot = any(
            imp.is_from and imp.module == '__future__'
            and any(n == 'annotations' for n, _ in imp.names)
            for imp in future_imports
        )
        if not has_future_annot:
            # Check if there's already a __future__ import to extend.
            added = False
            for i, imp in enumerate(future_imports):
                if imp.is_from and imp.module == '__future__':
                    new_names = list(imp.names) + [('annotations', None)]
                    future_imports[i] = _Import(
                        alias=None,
                        is_from=True,
                        level=0,
                        module='__future__',
                        names=_sort_imported_names(new_names),
                        raw=imp.raw,
                    )
                    added = True
                    break
            if not added:
                future_imports.append(_Import(
                    alias=None,
                    is_from=True,
                    level=0,
                    module='__future__',
                    names=[('annotations', None)],
                    raw='',
                ))

        # Ensure ``TYPE_CHECKING`` is imported from ``typing``.
        # Check if TYPE_CHECKING is already imported.
        tc_already_imported = False
        for imp in external:
            if imp.is_from and not imp.is_relative and imp.module == 'typing':
                for name, _ in imp.names:
                    if name == 'TYPE_CHECKING':
                        tc_already_imported = True
                        break
        if not tc_already_imported:
            # Add TYPE_CHECKING to the existing ``from typing import ...`` or
            # create a new one.
            added = False
            for i, imp in enumerate(external):
                if imp.is_from and not imp.is_relative and imp.module == 'typing':
                    new_names = list(imp.names) + [('TYPE_CHECKING', None)]
                    external[i] = _Import(
                        alias=None,
                        is_from=True,
                        level=0,
                        module='typing',
                        names=_sort_imported_names(new_names),
                        raw=imp.raw,
                    )
                    added = True
                    break
            if not added:
                external.append(_Import(
                    alias=None,
                    is_from=True,
                    level=0,
                    module='typing',
                    names=[('TYPE_CHECKING', None)],
                    raw='',
                ))
                external.sort(key=_external_sort_key)

    # Sort imports by their natural order — TYPE_CHECKING imports are
    # interleaved at their correct dot-level position and wrapped in
    # ``if TYPE_CHECKING:`` blocks during reconstruction.
    external.sort(key=_external_sort_key)
    relative.sort(key=_relative_sort_key)

    # --- Phase 6: Reconstruct the import block ---
    new_lines: List[str] = []
    for imp in future_imports:
        new_lines.append(_format_import(imp))
    if future_imports and (external or relative):
        new_lines.append('')

    def _emit_imports(imps: List[_Import]) -> None:
        """Append formatted import lines, grouping consecutive
        TYPE_CHECKING imports under a single ``if TYPE_CHECKING:`` block."""
        in_tc_block = False
        for imp in imps:
            if imp.type_checking:
                if not in_tc_block:
                    new_lines.append('if TYPE_CHECKING:')
                    in_tc_block = True
                new_lines.append('    ' + _format_import(imp))
            else:
                in_tc_block = False
                new_lines.append(_format_import(imp))

    _emit_imports(external)
    if external and relative:
        new_lines.append('')
    _emit_imports(relative)

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

    # If ``from __future__ import annotations`` was added (or was already
    # present), unquote string-literal annotations (e.g. ``'DicomPatient'``
    # → ``DicomPatient``).
    if has_tc_imports:
        new_source = _unquote_annotations(new_source)

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

    try:
        ast.parse(source)
    except SyntaxError as exc:
        print(f'  Syntax error in {path} (line {exc.lineno}): {exc.msg}')
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
        print('All imports are sorted and clean.')
    elif args.fix:
        print(f'Fixed imports in {total} file(s) total.')
    else:
        print(f'Found issues in {total} file(s). Re-run with --fix to apply.')

    sys.exit(0 if (total == 0 or args.fix) else 1)


if __name__ == '__main__':
    main()

import ast
from functools import wraps
import inspect
import numpy as np
import textwrap
import torch
from typing import Any, Callable, Dict, List, Tuple

from ..typing import Number
from .python import isinstance_generic, version

class CallVisitor(ast.NodeVisitor):
    def __init__(
        self,
        inner_fn: Callable):
        self.__inner_fn = inner_fn
        self.__args = []
        self.__kwargs = []
    
    @property
    def args(self) -> List[str]:
        return self.__args
        
    @property
    def kwargs(self) -> List[str]:
        return self.__kwargs

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.__inner_fn.__name__:
            for a in node.args:
                if isinstance(a, ast.Starred):
                    self.__args.append('args')
                else:
                    self.__args.append(ast.unparse(a))
            for k in node.keywords:
                if k.arg is None:
                    self.__kwargs.append('kwargs')
                else:
                    self.__kwargs.append(k.arg)

def alias_kwargs(
    *aliases: Tuple[str, str],
    ) -> Callable:
    alias_map = dict(aliases)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for shortcut, full_name in alias_map.items():
                if shortcut in kwargs:
                    kwargs[full_name] = kwargs.pop(shortcut)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def arg_to_list(
    arg: Any | None,
    types: Any | List[Any],     # Check if 'arg' matches any of these types.
    broadcast: int = 1,         # Expand a match to multiple elements, e.g. None -> [None, None, None].
    exceptions: Any | List[Any] | None = None,
    iter_types: Any | List[Any] | None = None,   # If 'arg' is one of these types, convert to list (e.g. list(obj)) instead of wrapping (e.g. [obj]).
    literals: Dict[Any, List[Any]] | None = None,   # Check if 'arg' matches any of these literal values.
    out_type: Any | None = None,    # Convert a match to a different output type.
    return_expanded: bool = False,   # Return whether the match was successful.
    ) -> List[Any]:
    # Convert types to list.
    if not isinstance(types, list) and not isinstance(types, tuple):
        types = [types]
    if exceptions is not None and not isinstance(exceptions, list) and not isinstance(exceptions, tuple):
        exceptions = [exceptions]
    if iter_types is not None and not isinstance(iter_types, list) and not isinstance(iter_types, tuple):
        iter_types = [iter_types]

    # Check exceptions.
    if exceptions is not None:
        for e in exceptions:
            if isinstance(arg, type(e)) and arg == e:
                if return_expanded:
                    return arg, False
                else:
                    return arg
    
    # Check literal matches.
    if literals is not None:
        for k, v in literals.items():
            if isinstance(arg, type(k)) and arg == k:
                arg = v

                # If arg is a function, run it now. This means the function
                # is not evaluated every time 'arg_to_list' is called, only when
                # the arg matches the appropriate literal (e.g. 'all').
                if isinstance(arg, Callable):
                    arg = arg()

                if return_expanded:
                    return arg, False
                else:
                    return arg

    # Check iterable types.
    if iter_types is not None:
        for t in iter_types:
            if isinstance(arg, t):
                arg = list(arg)
                if return_expanded:
                    return arg, False
                else:
                    return arg

    # Check types.
    expanded = False
    for t in types:
        if isinstance_generic(arg, t):
            expanded = True
            arg = [arg] * broadcast
            break
        
    # Convert to output type.
    if expanded and out_type is not None:
        arg = [out_type(a) for a in arg]

    if return_expanded:
        return arg, expanded
    else:
        return arg

def bubble_args(*inner_fns: Callable) -> Callable:
    if not version(gte='3.9'):
        # Ast unparse is not available in Python < 3.9.
        return lambda f: f

    def change_outer_fn_sig(outer_fn: Callable) -> Any:
        # Load params.
        outer_sig = inspect.signature(outer_fn)
        outer_params = dict(outer_sig.parameters)
        outer_params_args = dict((k, v) for k, v in outer_params.items() if v.default is inspect.Parameter.empty)
        outer_params_kwargs = dict((k, v) for k, v in outer_params.items() if v.default is not inspect.Parameter.empty)

        # Bubble some args from the inner function up to the outer function signature.
        bubbled_args = {}
        bubbled_kwargs = {}
        for f in inner_fns:
            inner_params = dict(inspect.signature(f).parameters)
            inner_args, inner_kwargs = get_inner_args(outer_fn, f)
            inner_params_args = dict((k, v) for k, v in inner_params.items() if v.default is inspect.Parameter.empty)
            inner_params_kwargs = dict((k, v) for k, v in inner_params.items() if v.default is not inspect.Parameter.empty)
            if 'args' in outer_params and 'args' in inner_args:
                for k, v in inner_params_args.items():
                    if k not in inner_args:  # I.e. not already passed by inner call.
                        bubbled_args[k] = v
            if 'kwargs' in outer_params and 'kwargs' in inner_kwargs:
                for k, v in inner_params_kwargs.items():
                    if k not in inner_kwargs:  # I.e. not already passed by inner call.
                        bubbled_kwargs[k] = v
                    
        # Create final signature.
        outer_params_args = dict((k, v) for k, v in outer_params_args.items() if k not in ['args', 'kwargs'])
        args = {}
        args = args | outer_params_args
        args = args | bubbled_args
        kwargs = {}
        kwargs = kwargs | outer_params_kwargs
        kwargs = kwargs | bubbled_kwargs
        # Sort alphabetically, but keyword-only params must be last.
        kw_only_kwargs = dict(sorted((k, v) for k, v in kwargs.items() if v.kind is inspect.Parameter.KEYWORD_ONLY))
        other_kwargs = dict(sorted((k, v) for k, v in kwargs.items() if v.kind is not inspect.Parameter.KEYWORD_ONLY))
        kwargs = other_kwargs | kw_only_kwargs
        params = args | kwargs
        
        outer_fn.__signature__ = outer_sig.replace(parameters=params.values())
        return outer_fn

    return change_outer_fn_sig

# Expands a single arg to a list of that type, e.g. 1 -> [1] if the 'arg' matches one of the 'types'.
# Converts directly to a list if the 'arg' is one of the 'iter_types'.
def expand_range_arg(
    arg: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor,
    dim: int = 3,   # Could be 2/3 for spatial or 1 for intensity.
    negate_lower: bool = False,
    vals_per_dim: int = 2,
    ) -> Tuple[Number, ...]:
    if isinstance(arg, (int, float)):
        arg = (-arg if negate_lower else arg, arg) * (vals_per_dim // 2) * dim
    elif isinstance(arg, (list, tuple, np.ndarray, torch.Tensor)):
        arg = tuple(np.array(arg).flatten())
        if len(arg) == vals_per_dim // 2:
            arg = arg * 2 * dim
        elif len(arg) == vals_per_dim:
            arg = arg * dim           
    return arg

# Expands an arg to the required length based on 'dim'.
def get_inner_args(
    outer_fn: Callable,
    inner_fn: Callable,
    ) -> Tuple[List[str], List[str]]:
    source = textwrap.dedent(inspect.getsource(outer_fn))
    tree = ast.parse(source)
    visitor = CallVisitor(inner_fn)
    visitor.visit(tree)
    return visitor.args, visitor.kwargs

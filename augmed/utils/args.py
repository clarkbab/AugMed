import ast
from functools import wraps
import inspect
import numpy as np
import textwrap
import torch
from typing import Any, Callable, Dict, List, Literal, Tuple

from ..typing import Number
from .python import isinstance_generic, version

# Does 'arg' have a value?
def arg_default(
    arg: Any | List[Any] | None,
    return_arg: Any | None,  # Return this value if at least one arg is not None.
    default: Any,    # Return this value is all args are None.
    ) -> Any:
    if not isinstance(arg, list) and not isinstance(arg, tuple):
        args = [arg]
    else:
        args = arg
    all_none = True
    for a in args:
        if a is not None:
            all_none = False
            break
    if all_none:
        return default
    else:
        return return_arg

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

# Checks if 'arg' matches any of the specified types and ensures that the outpu
# is a list of this type - e.g. converts a single str to a List[str].
def arg_to_list(
    arg: Any | None,
    types: Any | List[Any],
    broadcast: int = 1,             # Expand a list of length 1 to length 'broadcast'.
    literals: Dict[Any, List[Any]] | None = None,   # Some args (e.g. 'all') should behave differently.
    out_type: Any | None = None,    # Ensure that the output list of type 'out_type'.
    return_matched: bool = False,   # Returns whether 'arg' matched any 'types'.
    ) -> List[Any] | Any:
    # Convert types to list.
    if not isinstance(types, list) and not isinstance(types, tuple):
        types = [types]
    
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

                if return_matched:
                    return arg, False
                else:
                    return arg

    # Check types.
    matched = False
    for t in types:
        if isinstance_generic(arg, t):
            matched = True
            arg = [arg] * broadcast
            break
        
    # Convert to output type.
    if matched and out_type is not None:
        arg = [out_type(a) for a in arg]

    if return_matched:
        return arg, matched
    else:
        return arg

def assert_2d(
    data: np.ndarray | torch.Tensor,
    message: str = "Data must be 2D.",
    ) -> None:
    assert data.ndim == 2, message

# Expands an arg to a range - used for random transforms to define the parameter ranges.
# E.g. arg=0.1, dim=3 -> (-0.1, 0.1, -0.1, 0.1, -0.1, 0.1) when negate_lower=True, vals_per_dim=2.
# E.g. arg=(0.8, 1.2), dim=3 -> (0.8, 1.2, 0.8, 1.2, 0.8, 1.2) when negate_lower=False, vals_per_dim=2.
# Remove between 10-20 voxels from each end of each axis of the image.
# E.g. arg=(10, 20), dim=3, vals_per_dim=4 -> (10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20) when negate_lower=False.
def assert_3d(
    data: np.ndarray | torch.Tensor,
    message: str = "Data must be 3D.",
    ) -> None:
    assert data.ndim == 3, message

# Expands an arg to the required length based on 'dim'.
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

def expand_range_arg(
    arg: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor,
    check_range: Literal['<', '<=', '>', '>='] | None = None,
    dim: int = 3,   # Could be 2/3 for spatial or 1 for intensity.
    negate_lower: bool = False,
    vals_per_dim: int = 2,
    ) -> Tuple[Number, ...]:
    # Expand to the full number of args.
    if isinstance(arg, (int, float)):
        arg = (-arg if negate_lower else arg, arg) * (vals_per_dim // 2) * dim
    elif isinstance(arg, (list, tuple, np.ndarray, torch.Tensor)):
        arg = tuple(np.array(arg).flatten())
        if len(arg) == vals_per_dim // 2:
            arg = arg * 2 * dim
        elif len(arg) == vals_per_dim:
            arg = arg * dim

    # Check ranges within the expanded args.
    if check_range is not None:
        range_check_fns = {
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
        }
        range_check_fn = range_check_fns[check_range]
        pairs_per_dim = vals_per_dim // 2
        for d in range(dim):
            for p in range(pairs_per_dim):
                low = arg[d * vals_per_dim + p * 2]
                high = arg[d * vals_per_dim + p * 2 + 1]
                if not range_check_fn(low, high):
                    raise ValueError(f"Range check '{check_range}' failed for dim {d}, pair {p}: ({low}, {high}).")
    return arg

def get_inner_args(
    outer_fn: Callable,
    inner_fn: Callable,
    ) -> Tuple[List[str], List[str]]:
    source = textwrap.dedent(inspect.getsource(outer_fn))
    tree = ast.parse(source)
    visitor = CallVisitor(inner_fn)
    visitor.visit(tree)
    return visitor.args, visitor.kwargs

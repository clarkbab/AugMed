from functools import wraps
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Tuple

from ..typing import Number
from .conversion import to_list, to_tuple
from .python import isinstance_generic

def alias_kwargs(
    aliases: Tuple[str, str] | List[Tuple[str, str]],
    ) -> Callable:
    aliases = arg_to_list(aliases, tuple)
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

# Expands a single arg to a list of that type, e.g. 1 -> [1] if the 'arg' matches one of the 'types'.
# Converts directly to a list if the 'arg' is one of the 'iter_types'.
def arg_to_list(
    arg: Any | None,
    types: Any | List[Any],     # Check if 'arg' matches any of these types.
    broadcast: int = 1,         # Expand a match to multiple elements, e.g. None -> [None, None, None].
    exceptions: Any | List[Any] | None = None,
    iter_types: Any | List[Any] | None = None,   # If 'arg' is one of these types, convert directly to a list (if not already).
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
                arg = to_list(arg)
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

# Expands an arg to the required length based on 'dim'.
def expand_range_arg(
    arg: Number | Tuple[Number, ...] | np.ndarray | torch.Tensor,
    dim: int = 3,   # Could be 2/3 for spatial or 1 for intensity.
    negate_lower: bool = False,
    vals_per_dim: int = 2,
    ) -> Tuple[Number, ...]:
    if isinstance(arg, (int, float)):
        arg = (-arg if negate_lower else arg, arg) * (vals_per_dim // 2) * dim
    elif isinstance(arg, (list, tuple, np.ndarray, torch.Tensor)):
        arg = to_tuple(arg)
        if len(arg) == vals_per_dim // 2:
            arg = arg * 2 * dim
        elif len(arg) == vals_per_dim:
            arg = arg * dim           
    return arg

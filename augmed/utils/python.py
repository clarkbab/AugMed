import numpy as np
import sys
import torch
from typing import Any, List, Literal, Tuple, Union, get_args, get_origin

from ..typing import Image, Points

def get_group_device(
    data: Image | Points | List[Image | Points],
    device: torch.device | None = None,
    ) -> torch.device:
    if device is not None:
        return device

    from .args import arg_to_list
    data = arg_to_list(data, (np.ndarray, torch.Tensor))
    if isinstance(data[0], torch.Tensor):
        return data[0].device

    return torch.device('cpu')

def is_generic(t: Any) -> bool:
    return get_origin(t) is not None

def is_windows() -> bool:
    return 'win' in sys.platform

def isinstance_generic(
    a: Any,
    t: Any,
    ) -> bool:
    # Checks if 'a' is of type 't' for generic (e.g. List[], Dict[]) and
    # non-generic types.
    if t is None:
        return a is None
    if not is_generic(t):
        return isinstance(a, t)
    
    # Check main type - e.g. 'list' for List[str], or 'union' for Union[str, int].
    main_type = get_origin(t)

    if main_type is Literal:
        # Check for literal matches.
        literals = get_args(t)
        for l in literals:
            if a == l:
                return True
        return False
    
    if main_type is Union:
        # Check for any matching subtype.
        subtypes = get_args(t)
        for s in subtypes:
            if isinstance_generic(a, s):
                return True
        return False
    
    # If not a Union main type, then main type must match.
    if not isinstance(a, main_type):
        return False
    
    if main_type in (list, CSequence):
        # For iterable main types (one subtype only - e.g. List[str] or List[int]),
        # check that all elements in 'a' match the required subtype.
        subtype = get_args(t)[0]
        for ai in a:
            if not isinstance_generic(ai, subtype):
                return False
    elif main_type in (dict,):
        # For dict main types (key/value subtypes - e.g. Dict[str, int]),
        # check that all keys/values in 'a' match the required subtypes.
        k_subtype, v_subtype = get_args(t)
        for k, v in a.items():
            if not isinstance_generic(k, k_subtype) or not isinstance(v, v_subtype):
                return False
            
    return True

def version(
    gte: str | None = None,
    ) -> Tuple[int, int, int] | bool:
    info = sys.version_info
    if gte is not None:
        res = [int(x) for x in gte.split('.')]
        if len(res) == 1:
            gte_major, gte_minor, gte_micro = res[0], -1, -1
        elif len(res) == 2:
            gte_major, gte_minor, gte_micro = res[0], res[1], -1
        elif len(res) == 3:
            gte_major, gte_minor, gte_micro = res[0], res[1], res[2]
        if info.major > gte_major or (info.major == gte_major and info.minor > gte_minor) or (info.major == gte_major and info.minor == gte_minor and info.micro >= gte_micro):
            return True
        else:
            return False
    version = (info.major, info.minor, info.micro)
    return version

import ast
import inspect
import sys
import textwrap
from typing import Any, Callable, List, Literal, Tuple, Union, get_args, get_origin

from .misc import is_windows

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

def bubble_args(*inner_fns: Callable) -> Callable:
    if is_windows() or not version(gte='3.9'):
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

def get_inner_args(
    outer_fn: Callable,
    inner_fn: Callable,
    ) -> Tuple[List[str], List[str]]:
    source = textwrap.dedent(inspect.getsource(outer_fn))
    tree = ast.parse(source)
    visitor = CallVisitor(inner_fn)
    visitor.visit(tree)
    return visitor.args, visitor.kwargs

def is_generic(t: Any) -> bool:
    return get_origin(t) is not None

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

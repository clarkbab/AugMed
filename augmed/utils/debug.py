from typing import Any


def from_desc(desc: str) -> Any:
    """Reconstruct an augmed transform instance from its string description.

    Typical usage in a notebook after logging a transform::

        t = from_desc("Crop(device=None, dim=3, centre=(0.5, 0.5, 0.5), ...)")

    The description must be a valid Python expression relative to the
    ``augmed`` namespace (i.e. the format produced by each transform's
    ``__str__`` method).
    """
    import augmed
    namespace = vars(augmed)
    return eval(desc, namespace)

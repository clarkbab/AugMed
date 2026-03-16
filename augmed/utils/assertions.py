from ..typing import Orientation

def assert_orientation(
    orientation: Orientation,
    ) -> None:
    orientations = {'LAI', 'LAS', 'LPI', 'LPS', 'RAI', 'RAS', 'RPI', 'RPS'}
    if orientation not in orientations:
        raise ValueError(f"Invalid orientation '{orientation}'. Must be one of {orientations}.")

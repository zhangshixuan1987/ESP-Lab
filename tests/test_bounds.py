import numpy as np


def test_lat_bounds_clipped():
    vals = np.array([89, 87, 85])
    mid = 0.5 * (vals[1:] + vals[:-1])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    bounds = np.concatenate([[first], mid, [last]])
    bounds = np.clip(bounds, -90.0, 90.0)
    assert bounds.min() >= -90.0
    assert bounds.max() <= 90.0
    assert len(bounds) == len(vals) + 1

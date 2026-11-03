"""Rotation-angle conventions (angle_range_from_frames)."""
import numpy as np
import pytest

from diffractom import angle_range_from_frames


def test_angle_range_from_frames():
    om = np.arange(3001) * 0.12                       # DanMAX-like: start labels, one extra point, encoder wrap
    om_wrapped = np.where(om >= 180, om - 360, om)
    for o in (om, om_wrapped):
        rng, cen = angle_range_from_frames(o, "start", frames_per_bin=6, n_frames=3000)
        assert np.allclose(rng, [0, 360]) and np.isclose(cen[0], 0.36) and len(cen) == 500
    for label, off in (("start", 0.0), ("center", 0.25), ("end", 0.5)):
        rng, cen = angle_range_from_frames(np.arange(360) * 0.5 + off, label)
        assert np.allclose(rng, [0, 180]) and np.allclose(cen[:2], [0.25, 0.75])
    rng, cen = angle_range_from_frames(-np.arange(10) * 1.0, "start")      # decreasing rotation
    assert np.allclose(rng, [0, -10]) and np.isclose(cen[0], -0.5)
    with pytest.raises(ValueError):
        angle_range_from_frames(np.arange(10.0), "middle")
    with pytest.raises(ValueError):
        angle_range_from_frames(np.r_[0, 1, 2, 4, 5.0], "start")              # a missing frame

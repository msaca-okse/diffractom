"""segment_coverage on a synthetic detector sampled finely in (2theta, eta)."""
import numpy as np

from diffractom.utils import segment_coverage


def _pixels(n_tt=400, n_eta=720):
    tt_edges = np.radians(np.linspace(4.0, 12.0, n_tt + 1))
    eta_edges = np.linspace(-180.0, 180.0, n_eta + 1)
    tt = 0.5 * (tt_edges[1:] + tt_edges[:-1])
    eta = 0.5 * (eta_edges[1:] + eta_edges[:-1])
    TT, ETA = np.meshgrid(tt, eta, indexing="ij")
    om = (np.cos(tt_edges[:-1]) - np.cos(tt_edges[1:]))[:, None] * np.radians(np.diff(eta_edges))[None, :]
    return TT, ETA, np.broadcast_to(om, TT.shape)


def test_full_and_partial_coverage():
    TT, ETA, om = _pixels()
    rings = np.radians([[5.0, 6.0], [9.0, 10.0]])
    eta_edges = np.linspace(-180.0, 180.0, 37)
    valid = np.ones(TT.shape, dtype=bool)
    cov = segment_coverage(TT, ETA, om, valid, rings, eta_edges)
    assert cov.shape == (36, 2)
    np.testing.assert_allclose(cov, 1.0, atol=1e-9)
    valid[(ETA > 0) & (ETA < 5)] = False                          # half of the eta bin [0, 10)
    valid[(TT > np.radians(9.5)) & (ETA < -170)] = False          # upper half of ring 2 in [-180, -170)
    cov = segment_coverage(TT, ETA, om, valid, rings, eta_edges)
    np.testing.assert_allclose(cov[18], 0.5, atol=1e-9)
    w = np.cos(np.radians(9.0)) - np.cos(np.radians(9.5))
    np.testing.assert_allclose(cov[0, 1], w / (np.cos(np.radians(9.0)) - np.cos(np.radians(10.0))), atol=1e-9)
    np.testing.assert_allclose(np.delete(cov, [0, 18], axis=0), 1.0, atol=1e-9)

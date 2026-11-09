"""Exact cell averages on the GPU: the cells fill orientation space and add up to the powder spectrum."""
import itertools

import numpy as np
import pyopencl as cl
import pytest
from scipy.spatial.transform import Rotation

from diffractom.operators.bragg_cell_integration import CellIntegrator, VoronoiCells, _amplitude, beam_directions


def cubic_rotations():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, p in enumerate(perm):
                m[i, p] = signs[i]
            if np.linalg.det(m) > 0:
                ops.append(m)
    return Rotation.from_matrix(np.array(ops))


def fcc_table(a=3.6):
    hkl = [h for h in itertools.product(range(-2, 3), repeat=3) if sum(x * x for x in h) in (3, 4) and len({x % 2 for x in h}) == 1]
    g = np.array(hkl, dtype=float) / a
    return {"g_vecs": g, "d": 1 / np.linalg.norm(g, axis=1), "F2": np.ones(len(g)), "V": a ** 3}


@pytest.fixture(scope="module")
def cl_env():
    ctx = cl.create_some_context(interactive=False)
    return ctx, cl.CommandQueue(ctx)


@pytest.mark.parametrize("partner", [None, Rotation.from_euler("z", 180, degrees=True)])
def test_cells_add_up_to_the_powder(cl_env, partner):
    ctx, queue = cl_env
    table = fcc_table()
    symmetry = cubic_rotations()
    cells = VoronoiCells(Rotation.random(12, random_state=3), symmetry, partner)
    radii = cells.piece_radii(n_samples=4_000_000)
    h = 2e-3
    lam = np.arange(1.5, 4.0, h) + h / 2
    integ = CellIntegrator(table, beam_directions([0.0, 37.0]), lam, ctx, queue, h_arc_deg=0.5)
    total, volume = 0.0, 0.0
    for k in range(cells.n_cells):
        S, vol = integ.cell_density(cells, k, radii[k])
        total = total + vol * S.sum(0)
        volume += vol
    assert len(symmetry) * volume == pytest.approx(1.0, abs=2e-3)
    powder = np.zeros_like(lam)
    for d, F2 in zip(table["d"], table["F2"]):
        inside = lam < 2 * d - 2 * h
        powder[inside] += _amplitude(lam[inside], d, F2, table["V"]) / (4 * d)
    # away from the absorption edges, where the bins straddle the step
    keep = np.all(np.abs(lam[:, None] - 2 * np.unique(np.round(table["d"], 6))[None]) > 4 * h, axis=1)
    for spectrum in len(symmetry) * total:
        assert np.linalg.norm(spectrum[keep] - powder[keep]) / np.linalg.norm(powder[keep]) < 5e-3

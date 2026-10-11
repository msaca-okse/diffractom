"""FISTASum: the volume-fraction penalty decides the split between operators the data cannot separate."""
import numpy as np

from diffractom.optimization.fista_sum import FISTASum


class _Dense:
    def __init__(self, A):
        self.A = A
        self.coeff_shape = (A.shape[1], 1, 1)

    def direct(self, x):
        return (self.A @ x.ravel()).reshape(-1, 1, 1).astype(np.float32)

    def adjoint(self, y):
        return (self.A.T @ y.ravel()).reshape(self.coeff_shape).astype(np.float32)

    def support_mask(self):
        return np.ones((1, 1), bool)


def test_volume_fractions_split_identical_operators():
    A = np.random.default_rng(0).random((40, 6)).astype(np.float32)
    ops = [_Dense(A), _Dense(A.copy())]
    b = 2 * ops[0].direct(np.ones((6, 1, 1), np.float32))
    L = 2 * np.linalg.norm(A, 2) ** 2
    for target in (None, [0.75, 0.25]):
        solver = FISTASum(ops, L=L, volume_fractions=target)
        xs = solver.run([np.zeros((6, 1, 1), np.float32)] * 2, b, niter=3000, verbose=0)
        S = np.array([x.sum() for x in xs])
        assert np.linalg.norm(solver.direct(xs) - b) < 1e-3 * np.linalg.norm(b)
        np.testing.assert_allclose(S / S.sum(), [0.5, 0.5] if target is None else target, atol=2e-3)


def test_support_mask():
    A = np.random.default_rng(1).random((30, 4)).astype(np.float32)
    op = _Dense(A)
    op.coeff_shape = (2, 1, 2)
    op.support_mask = lambda: np.ones((1, 2), bool)
    b = op.direct(np.ones((2, 1, 2), np.float32))
    x = FISTASum([op], L=np.linalg.norm(A, 2) ** 2, support=np.array([[1, 0]])).run([np.zeros((2, 1, 2), np.float32)], b, niter=50, verbose=0)[0]
    assert np.all(x[:, :, 1] == 0) and np.any(x[:, :, 0] > 0)

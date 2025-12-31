import numpy as np
from package.texture_tomography.operators.operator_memory_model import OperatorMemoryModel


class FISTAMemoryModel:
    """
    Deterministic memory model for FISTAOpenCL.

    This class does NOT allocate GPU memory.
    It only accounts for memory usage via MemoryCounter.
    """

    def __init__(self, fista, mem, op_model):
        """
        Parameters
        ----------
        fista : FISTAOpenCL
            The real FISTA operator
        mem : MemoryCounter
            Shared memory counter
        op_model : OperatorMemoryModel
            ALREADY INITIALIZED operator memory model
        """
        self.fista = fista
        self.mem = mem
        self.op_model = op_model

    # ------------------------------------------------------------------
    # persistent buffers (allocated once per run)
    # ------------------------------------------------------------------
    def model_persistent_buffers(self, x_shape, Ax_shape):
        """
        Model buffers allocated once at the beginning of FISTA.run().
        """
        mem = self.mem

        # primal / extrapolation buffers
        mem.alloc("fista.x", x_shape, np.float32)
        mem.alloc("fista.y", x_shape, np.float32)
        mem.alloc("fista.x_old", x_shape, np.float32)
        mem.alloc("fista.v", x_shape, np.float32)

        # gradient buffer (allocated after first adjoint, but persistent)
        mem.alloc("fista.grad", x_shape, np.float32)

        # forward / residual buffers
        mem.alloc("fista.Ax", Ax_shape, np.float32)
        mem.alloc("fista.r", Ax_shape, np.float32)

        # TV prox buffers (only if needed)
        if self.fista.prox_kind == "nonneg_tv":
            for k in ["y", "gx", "gy", "px", "py", "div"]:
                mem.alloc(f"fista.tv.{k}", x_shape, np.float32)

    # ------------------------------------------------------------------
    # one iteration (this is where PEAK memory happens)
    # ------------------------------------------------------------------
    def model_one_iteration(self):
        """
        Model one FISTA iteration.

        Peak memory = persistent buffers
                    + one forward
                    + one adjoint
        """
        mem = self.mem

        mem.enter_scope("fista.iteration")

        # ---- Ax_new = A(y) ----
        self.op_model.model_direct_cl()

        # ---- grad_new = A*(r) ----
        self.op_model.model_adjoint_cl()

        # all temporaries freed here
        mem.exit_scope()

    # ------------------------------------------------------------------
    # full run
    # ------------------------------------------------------------------
    def model_run(self, x_shape, Ax_shape, niter=1):
        """
        Model a full FISTA run.

        NOTE:
        - niter=1 is sufficient to capture PEAK memory.
        """
        # ---- persistent allocations ----
        self.model_persistent_buffers(x_shape, Ax_shape)

        # ---- iterations ----
        for _ in range(niter):
            self.model_one_iteration()

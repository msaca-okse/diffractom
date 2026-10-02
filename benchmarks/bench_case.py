"""
One benchmark case, run in a fresh process against one diffractom source tree.

    PYTHONPATH=<tree>/src python bench_case.py '<case json>'

Prints one line `RESULT {json}`. The problem is synthetic (no data files): K random orientations
with Gaussian width sigma, an Nx x Ny grid with My = Nx translations, N_Omega rotation steps,
N_eta azimuthal bins and one aluminium reflection per ring. Speed and memory depend only on these
sizes, not on the data values.

Modes
    gpu      coefficients and data on the GPU (pyopencl arrays); works with every version
    stream   NumPy in and out, coefficients streamed through the GPU (reserve_coefficient_arrays=0);
             versions with op.coeff_shape (from 14ec22e)

Measured: operator build (incl. the PF matrix), one forward, one adjoint, one power iteration
(for L), FISTA-Huber (nonneg, library defaults) per iteration, each with the peak GPU memory of
this process during that phase (nvidia-smi, sampled every 50 ms; includes the ~0.3-0.5 GB CUDA
context) and the peak host RSS of the whole process.
"""
import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

AL_HKL = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0), (3, 3, 1), (4, 2, 0), (4, 2, 2),
          (5, 1, 1), (4, 4, 0), (5, 3, 1), (6, 0, 0), (6, 2, 0), (5, 3, 3), (6, 2, 2)]


class GpuMemorySampler:
    """Peak GPU memory (MiB) of this process, from `nvidia-smi --query-compute-apps` every 50 ms."""

    def __init__(self):
        self.samples = []  # (time, MiB)
        self.pid = str(os.getpid())
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == self.pid and parts[1].isdigit():
                self.samples.append((time.perf_counter(), int(parts[1])))

    def peak(self, t0=None, t1=None):
        time.sleep(0.15)  # let the last samples arrive
        m = [mb for t, mb in self.samples if (t0 is None or t >= t0) and (t1 is None or t <= t1 + 0.1)]
        return max(m) if m else None

    def stop(self):
        self.proc.terminate()


def main():
    case = json.loads(sys.argv[1])
    K, N, sigma_deg = int(case["K"]), int(case["N"]), float(case["sigma_deg"])
    n_omega, n_eta, n_rings = int(case["N_Omega"]), int(case["N_eta"]), int(case["rings"])
    mode, fista_iters, max_gb = case.get("mode", "gpu"), int(case.get("fista_iters", 3)), float(case.get("max_gb", 2.0))
    out = {"case": case, "status": "ok", "phases": {}}
    sampler = GpuMemorySampler()
    phases = out["phases"]

    def phase(name, fn):
        t0 = time.perf_counter()
        r = fn()
        t1 = time.perf_counter()
        phases[name] = {"seconds": t1 - t0, "_t": (t0, t1)}
        return r

    try:
        import diffractom
        import pyopencl as cl
        import pyopencl.array as clarray
        from scipy.spatial.transform import Rotation
        from diffractom import FISTAHuber, Grid, Material, SinglePhaseForwardOperator
        from diffractom.operators.single_phase_forward_operator import estimate_L_power
        out["diffractom_file"] = diffractom.__file__

        cfg = {"N_theta": n_rings, "N_Omega": n_omega, "N_Omega_subdivisions": 1, "angle_range": [0, 180],
               "Nx": N, "Ny": N, "My": N, "k_direction_0": [0, 0, 1], "p_direction_0": [1, 0, 0],
               "j_direction_0": [0, -1, 0], "detector_direction_origin": [0, -1, 0],
               "detector_direction_positive_90": [0, 0, -1], "cor_offset": 0, "N_eta": n_eta,
               "N_eta_subdivisions": 1, "eta_angle_range": [0, 360], "energy": 50}
        mat = Material.from_lattice_parameters(lattice_matrix=np.eye(3) * 4.0495, lattice_matrix_kind="direct",
                                               symmetry_group="cubic", wavelength_A=12.398 / 50.0,
                                               hkl_list=np.array(AL_HKL[:n_rings]))
        rots = Rotation.random(K, random_state=0).as_matrix()
        grid = Grid.from_rotation_matrices(rots, np.deg2rad(sigma_deg))

        kw = {"reserve_coefficient_arrays": 0} if mode == "stream" else {}
        op = phase("build", lambda: SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=max_gb,
                                                               verbose=False, normalized=True, **kw))
        q = op.queue
        out["K"], out["N_seg"] = int(op.K), int(op.N_seg)
        out["pf_mode"] = getattr(op, "pf_mode", "dense")  # versions before the sparse path: always dense
        out["pf_storage"] = getattr(op, "pf_storage", None)  # sparse modes, from 6a33ab7
        out["batches"], out["K_batch_max"] = len(op.batches), int(op.K_batch_max)
        new_layout = hasattr(op, "coeff_shape")
        cshape = (op.K, op.Ny, op.Nx) if new_layout else (op.Nx, op.Ny, op.K)
        corder = "C" if new_layout else "F"
        rng = np.random.default_rng(1)
        b_host = rng.random((op.N_Omega, op.My, op.N_seg), dtype=np.float32)

        if mode == "stream":
            if not new_layout:
                raise RuntimeError("stream mode needs a version with op.coeff_shape")
            from diffractom import estimate_L_power_streamed
            x_host = np.zeros(op.coeff_shape, np.float32)
            phase("forward", lambda: op.direct(x_host))
            phase("adjoint", lambda: op.adjoint(b_host))
            L = phase("power_iteration", lambda: estimate_L_power_streamed(op, niter=1, seed=0, verbose=0))
            solver = FISTAHuber(op, prox_kind="nonneg", L=1.1 * L)
            phase("fista", lambda: solver.run(x_host, b_host, niter=fista_iters, verbose=0))
        else:
            x = clarray.zeros(q, cshape, np.float32, order=corder)
            b = clarray.to_device(q, b_host)
            del b_host

            def run(f):
                r = f()
                q.finish()
                return r
            phase("forward", lambda: run(lambda: op.direct(x)))
            phase("adjoint", lambda: run(lambda: op.adjoint(b)))
            L = phase("power_iteration", lambda: estimate_L_power(op, niter=1, seed=0, verbose=0))
            solver = FISTAHuber(op, prox_kind="nonneg", L=1.1 * L)
            phase("fista", lambda: run(lambda: solver.run(x, b, niter=fista_iters, verbose=0)))
        phases["fista"]["per_iteration"] = phases["fista"]["seconds"] / fista_iters
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        oom = any(s in msg.upper() for s in ("MEM_OBJECT_ALLOCATION_FAILURE", "OUT_OF_RESOURCES", "OUT_OF_HOST_MEMORY",
                                              "MEMORYERROR", "OUT OF MEMORY"))
        out["status"] = "oom" if oom else "error"
        out["error"] = msg[:500]
        out["traceback"] = traceback.format_exc()[-2000:]
    for p in phases.values():
        t0, t1 = p.pop("_t")
        p["peak_gpu_mib"] = sampler.peak(t0, t1)
    out["peak_gpu_mib"] = sampler.peak()
    out["peak_host_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    sampler.stop()
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

# diffractom — Documentation

GPU-accelerated texture tomography reconstruction library.  
Reconstructs spatially-resolved orientation distribution functions (ODFs) from
synchrotron powder-diffraction data using pole-figure transforms and
parallel-beam projection on OpenCL.

---

## Package layout

```
diffractom/
├── operators/          # Forward / adjoint operators (GPU)
├── crystallography/    # Material, lattice, point-group helpers
├── utils/              # Orientation grid tree, GPU memory logger
└── optimization/       # FISTA solvers, proximal operators, TV
```

---

## Subpackages

### `operators`

| Module | Description |
|--------|-------------|
| `pfo_single_material.py` | `PFO_SINGLE` — single-material pole-figure ⊗ Radon operator. |
| `pfo_and_projection_batched_opencl.py` | `PFO_OPENCL_BATCHED` — multi-material operator with Gaussian 2θ convolution. |
| `create_pfo_matrix.py` | OpenCL kernel source strings and GPU wrapper functions for PF-matrix evaluation (dense & sparse). |
| `pfo_kernels.py` | Builds the main `.cl` program, typed `Kernels` dataclass, `build_all_opencl()` convenience. |
| `pfo_kernels.cl` | Raw OpenCL kernels: GEMM, transposes, slicing, scattering, Gaussian peak expansion. |

#### Key classes

**`PFO_SINGLE(cfg, material, grid, max_gb, ...)`**

Single-material forward operator.  Computes `data = PF @ Radon(coeffs)` and
its adjoint entirely on GPU.

* `direct(coeffs)` → allocating forward.
* `direct_cl(coeffs, data)` → in-place forward (accumulates into `data`).
* `adjoint(data)` → allocating adjoint.
* `adjoint_cl(data, coeffs)` → in-place adjoint.
* `free_memory()` → release GPU buffers.

**`PFO_OPENCL_BATCHED(cfg, materials, grids, two_thetas, max_gb, ...)`**

Multi-material operator.  Each material has its own orientation grid and set
of reflections.  Gaussian peak convolution maps per-peak PF values onto a
detector 2θ grid.

* Same `direct` / `adjoint` API as `PFO_SINGLE`.
* `set_peak_width(pw)` — update Gaussian broadening.
* `convolve_matrix_from_pf_batch(...)` — internal convolution step.

#### Module-level helpers

* `batched_gemm_clblast(queue, A3, B3, C3, R, M, K, N)` — batched GEMM via CLBlast (accumulate).
* `batched_gemm_adj_clblast(queue, Y3, BT3, X3, ...)` — batched adjoint GEMM (overwrite).
* `estimate_L_power(op, niter=20)` — Lipschitz constant via power iteration.

---

### `crystallography`

| Module | Description |
|--------|-------------|
| `material.py` | `Material` class — CIF parsing, XRD reflections, reciprocal-lattice vectors. |
| `lattice.py` | Direct and reciprocal lattice matrices for the 7 Bravais systems. |
| `point_groups.py` | Point-group symmetry rotations as `scipy.spatial.transform.Rotation` tuples. |

#### `Material`

Factory methods:

* `Material.from_cif(cif_path, wavelength_kev=..., min_two_theta=..., max_two_theta=...)`
* `Material.from_lattice_parameters(name, crystal_system, lattice_params, ...)`

Important methods:

* `compute_h_vectors()` — fill `h_vecs`, `h_vecs_normed`.
* `attach_point_group(point_group_map)` — attach symmetry operators.
* `filter_by_intensity(min_intensity)` / `filter_by_two_theta(tth_min, tth_max)`.
* `hkls()`, `two_theta()`, `d_spacings()`, `intensities()` — accessors.
* `summary()` — quick overview dict.

#### Lattice functions

All return `(A, B)` where `A` is the direct lattice matrix and
`B = 2π (A⁻¹)ᵀ` the reciprocal lattice matrix.

`cubic`, `tetragonal`, `orthorhombic`, `hexagonal`,
`trigonal_rhombohedral`, `monoclinic`, `triclinic`.

#### Point groups

Pre-defined as tuples of `Rotation` objects in `point_groups.py`:

`trivial`, `cyclic_2`, `orthorhombic`, `cyclic_3`, `trigonal`,
`cyclic_4`, `tetragonal`, `cyclic_6`, `hexagonal`, `tetrahedral`,
`cubic` (= `octahedral`).

---

### `utils`

| Module | Description |
|--------|-------------|
| `multiresolution_refiner.py` | `OrientationTree` + `OrientationNode` — hierarchical orientation grid. |
| `gpu_live_tracker.py` | `GPUMemoryLogger` — poll GPU memory usage in a background thread. |

#### `OrientationTree`

```python
tree = OrientationTree(sigma_levels=[0.15, 0.08, 0.04])
tree = OrientationTree.from_rotation_matrices(R_mats, sigma=0.15)
```

* `generate_children(parents, radius, stencil=6)` — refine selected nodes.
* `active_leaf_nodes()` — indices of active leaves (used as orientation basis).
* `rotations_at_level(level)`, `scores_at_level(level)`.
* `summary()`, `print_summary()`, `inspect_node(idx)`.

#### `GPUMemoryLogger`

```python
logger = GPUMemoryLogger(interval=0.2, gpu_id=0)
logger.start()
# ... GPU work ...
logger.stop()
t, mem = logger.as_arrays()
```

---

### `optimization`

| Module | Description |
|--------|-------------|
| `fista_opencl.py` | `FISTAOpenCL` — FISTA solver with proximal operators on GPU. |
| `fista_huber_opencl.py` | `FISTAHuberOpenCL` — FISTA with Huber loss (robust to outliers). |
| `prox.py` | `ProxKernels` + `prox_nonneg`, `prox_l1`, `prox_nonneg_l1`. |
| `prox_tv.py` | `TVProxKernels` + `prox_tv_nonneg_inplace` (Chambolle TV denoising). |

#### `FISTAOpenCL`

```python
solver = FISTAOpenCL(operator, prox_kind="nonneg_l1", lam=1e-3, L=L_est)
x_sol = solver.run(x0_gpu, b_gpu, niter=200, verbose=1)
```

Supported `prox_kind` values: `"nonneg"`, `"l1"`, `"nonneg_l1"`, `"nonneg_tv"`.

After `run()`, diagnostics are available in `solver.iter_stats` (list of dicts)
and `solver.final_stats`.

#### `FISTAHuberOpenCL`

Same API as `FISTAOpenCL` plus a `huber_delta` parameter that controls the
quadratic-to-linear transition of the Huber loss.

---

## Configuration dict (`cfg`)

Operators expect a `cfg` dict (or mapping) with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `Nx`, `Ny` | int | Spatial grid size (pixels). |
| `N_Omega` | int | Number of projection angles. |
| `N_eta` | int | Number of azimuthal detector bins. |
| `angle_range` | [float, float] | Min/max projection angle (degrees). |
| `wavelength` | float | X-ray energy (keV). |
| `detector_direction_origin` | [3] | Unit vector for η = 0. |
| `detector_direction_positive_90` | [3] | Unit vector for η = 90°. |
| `k_direction_0` | [3] | Sample rotation axis. |
| `p_direction_0` | [3] | Beam direction unit vector. |
| `N_Omega_subdivisions` | int | Sub-steps per Ω interval (default 1). |
| `peak_width` | float | Gaussian peak σ (only for batched operator). |

Any key can be overridden at construction time via `**kwargs`:
```python
op = PFO_SINGLE(cfg, material, grid, max_gb=2.0, N_Omega=100)
```

---

## Dependencies

* **PyOpenCL** — GPU compute
* **gratopy** — parallel-beam Radon transform
* **pyclblast** — batched GEMM on OpenCL
* **scipy** — `Rotation` for orientation math
* **pymatgen** + **ASE** — CIF parsing, XRD calculation
* **numpy**

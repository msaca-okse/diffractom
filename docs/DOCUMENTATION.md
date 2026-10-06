# diffractom documentation

diffractom reconstructs spatially resolved orientation distribution functions
(ODFs) from diffraction data. The ODF of each pixel is a non-negative
combination of K Gaussian kernels on an orientation grid, and the forward
operator combines a pole-figure transform with a parallel-beam Radon
transform. Everything that scales with the problem size runs in OpenCL.

For installation and a first example, see the [README](../README.md) and
`examples/example_diffractom_pipeline.ipynb`.

## Package layout

```
diffractom/
├── crystallography/   Material, reflections, lattices, point groups, space groups
├── operators/         Forward and adjoint operators (GPU)
├── optimization/      FISTA solvers, proximal operators, TV prox
└── utils/             Orientation grid, support masks, reinterpolation, GPU memory logger
```

The top-level package exports `Material`, `Grid`, `fov_support_mask`,
`SinglePhaseForwardOperator`, `MultiPhaseForwardOperator`,
`BulkTextureForwardOperator`, `MatrixTomographicOperator`,
`BraggEdgeTomographicOperator`, `FISTAHuber`, `FISTAL2`,
`estimate_L_power_streamed` and the reinterpolation functions (`build_interpolation_kernelSO3`, `interpolateSO3`,
`build_interpolation_kernelS2`, `interpolateS2`, `project_rotations_to_s2`).

## Array conventions

| Array | Shape | Memory order |
|-------|-------|--------------|
| Coefficients `x` | `(K, Ny, Nx)`: `x[k]` is the image of orientation k (rows y, columns x) | C |
| Data `b` (single-phase operator) | `(N_Omega, My, N_eta * N_rings)`, i.e. `(N_Omega, My, N_eta, N_rings)` flattened with rings fastest | C |
| Data `b` (multi-phase operator) | `(N_Omega, My, N_eta * N_theta)` with `N_theta` detector 2θ bins | C |

| Weights (FISTAHuber) | `N_eta * N_rings` elements, e.g. `(N_eta, N_rings)` | C |
| Support mask | `(Ny, Nx)`, boolean | C |

All arrays are float32 and C-contiguous; `op.coeff_shape` and `op.data_shape`
give the shapes. K is the number of active leaf nodes of the orientation grid.
The solvers and `direct` / `adjoint` take NumPy arrays (converted to
C-contiguous float32 if needed; the coefficients are then streamed through the
GPU, see below) or `pyopencl.array.Array` objects on the operator's queue
(`op.queue`), which must already be C-contiguous float32 of the right shape;
otherwise a `ValueError` or `TypeError` says what is wrong.

## Configuration (`cfg`)

The operators take the scan geometry as a plain dictionary, set in the script or
notebook (see `examples/example_diffractom_pipeline.ipynb`). Any key can be
overridden as a keyword argument of the operator, e.g.
`SinglePhaseForwardOperator(cfg, mat, grid, max_gb=2.0, N_Omega=100)`. Keys with a
default may be left out; `SinglePhaseForwardOperator` reads no other keys (an
older `j_direction_0` entry, for instance, is ignored).

| Key | Type | Description |
|-----|------|-------------|
| `Nx`, `Ny` | int | Reconstruction grid size in pixels (pixel size = translation step). |
| `My` | int | Number of translation steps (detector pixels of the sinogram). |
| `N_Omega` | int | Number of rotation steps. |
| `angle_range` | [float, float] | Rotation range in degrees. The rotation angles are the midpoints of `N_Omega` equal steps (with one subdivision). |
| `N_Omega_subdivisions` | int | Integration samples per rotation step in the pole-figure transform (default 1). |
| `N_eta` | int | Number of azimuthal bins. |
| `eta_angle_range` | [float, float] | Azimuthal range in degrees (default [0, 360]); bin midpoints are used with one subdivision. |
| `N_eta_subdivisions` | int | Integration samples per azimuthal bin (default 1). |
| `energy` | float | X-ray energy in keV. It sets the ring 2θ used for the detector geometry and should match the material's wavelength. |
| `k_direction_0` | [3] | Rotation axis. |
| `p_direction_0` | [3] | Beam direction. |
| `detector_direction_origin` | [3] | Direction of η = 0 on the detector. |
| `detector_direction_positive_90` | [3] | Direction of η = 90° on the detector. |
| `cor_offset` | float | Centre-of-rotation offset in pixels. |
| `peak_width` | float | Gaussian peak width in degrees 2θ (`MultiPhaseForwardOperator` only). |

`BulkTextureForwardOperator` has no tomographic part and does not use `Nx`,
`Ny`, `My` or `cor_offset`.

## `crystallography`

| Module | Contents |
|--------|----------|
| `material.py` | `Material`: lattice, space group, reflections, reciprocal-lattice vectors and point-group operators. |
| `cif_parser.py` | `parse_cif(path)`, a pure-Python CIF parser used by `Material.from_cif`. |
| `form_factors.py` | X-ray atomic form factors (Cromer-Mann): `form_factor`, `form_factor_array`. |
| `space_group_centering.py` | Space-group number/symbol table (`resolve_space_group`) and the integral (centering) reflection conditions. |
| `lattice.py` | Direct lattice matrices for the crystal systems (`cubic`, `tetragonal`, `orthorhombic`, `hexagonal`, `trigonal_rhombohedral`, `monoclinic`, `triclinic`) and `reciprocal_lattice(A)` = 2π (A⁻¹)ᵀ. |
| `point_groups.py` | Proper point groups as tuples of `scipy.spatial.transform.Rotation`: `trivial`, `cyclic_2`, `cyclic_3`, `cyclic_4`, `cyclic_6`, `orthorhombic`, `trigonal`, `tetragonal`, `hexagonal`, `tetrahedral`, `cubic` (= `octahedral`). |
| `neutron_material.py` | `NeutronMaterial`, a cubic material defined by neutron scattering parameters, for Bragg-edge modelling. |

### `Material`

Factory methods (all keyword arguments after the first):

```python
# From a CIF file: structure factors and intensities are computed
mat = Material.from_cif("Al.cif", wavelength_A=0.35,
                        min_two_theta=np.deg2rad(2), max_two_theta=np.deg2rad(15))

# From lattice parameters: no atomic basis, intensities 1 scaled by multiplicity
mat = Material.from_lattice_parameters(a=4.05, space_group_number=225, wavelength_A=0.35,
                                       hkl_list=[(1, 1, 1), (2, 0, 0), (2, 2, 0)])

# From a lattice matrix and a crystal system
mat = Material.from_lattice_parameters(lattice_matrix=A, lattice_matrix_kind="direct",
                                       symmetry_group="cubic", wavelength_A=0.35, hkl_list=hkls)

# From neutron scattering sites (for Bragg-edge imaging)
mat = Material.from_neutron_sites(sites, a=3.596, space_group_number=225)
```

- **Radiation:** give either `wavelength_A` (Å) or `energy_kev` (keV).
- **Reflections:** select them with `hkl_list`, a 2θ range
  (`min_two_theta` and `max_two_theta`, in radians) or a q range (`q_min`, `q_max`,
  in Å⁻¹, q = 2π/d).
- **Space group:** `space_group_number` or `symbol` sets it. Reflections
  forbidden by the lattice centering are then removed (`filter_extinct`).
  Glide and screw extinctions are not applied, and R-centred groups assume
  hexagonal axes.

Reflections are stored in the structured array `mat.reflections` with fields
`hkl`, `multiplicity`, `d_spacing`, `two_theta` (radians) and `intensity`.
Accessors: `hkls()`, `multiplicities()`, `two_theta()`, `d_spacings()`,
`intensities()`, `reflection(i)` and `summary()`. `filter_by_intensity` and
`filter_by_two_theta` remove reflections in place.

`compute_reflections` also computes the reciprocal-lattice vectors
(`compute_h_vectors`) and the point-group operators (`attach_point_group`),
which the operators read. The point group follows the crystal system:
triclinic → `trivial`, monoclinic → `cyclic_2`, orthorhombic, trigonal,
tetragonal, hexagonal and cubic → the group of the same name. Note that
`cyclic_2` rotates about z, while the conventional monoclinic unique axis is b.

## `operators`

All operators share the same interface:

- `direct(x)` returns a new data array: a NumPy array for a NumPy `x` (the
  coefficients are streamed to the GPU batch by batch), a pyopencl array for a
  pyopencl `x`;
- `direct_cl(x, b)` writes the result of the forward operator into `b`;
- `adjoint(b)` returns a new coefficient array, NumPy for a NumPy `b` (streamed
  back batch by batch), pyopencl for a pyopencl `b`;
- `adjoint_cl(b, x)` writes the result of the adjoint into `x`;
- `adjoint_batches_cl(b, out_batch, update)` computes the adjoint one orientation
  batch at a time into a batch-sized buffer and calls `update(k0, Kb)` after each
  batch (used by the fused FISTA update);
- `free_memory()` releases the operator's GPU buffers.

Each operator creates its own OpenCL context unless `ctx` and `queue` are
passed; select the device with the `PYOPENCL_CTX` environment variable.

| Module | Contents |
|--------|----------|
| `single_phase_forward_operator.py` | `SinglePhaseForwardOperator`, `estimate_L_power`, `group_reflections_into_rings`. |
| `multi_phase_forward_operator.py` | `MultiPhaseForwardOperator`. |
| `bulk_texture_forward_operator.py` | `BulkTextureForwardOperator`. |
| `matrix_tomographic_operator.py` | `MatrixTomographicOperator`. |
| `bragg_edge_tomographic_operator.py` | `BraggEdgeTomographicOperator`, `build_bragg_matrix_cpu`, `build_bragg_matrix_gpu`, `build_bragg_matrix_quadrature`. |
| `parallel_radon.py` | `ParallelRadon`, the parallel-beam Radon transform vectorised over orientations. |
| `create_pfo_matrix.py`, `pf_kernels.py`, `*.cl` | OpenCL programs for pole-figure matrix evaluation, batched GEMM and sparse products. |

### `SinglePhaseForwardOperator(cfg, material, grid, max_gb, ...)`

Computes `b = PF @ Radon(x)` for one material. Reflections with the same 2θ
form one ring, i.e. one data channel, so the data have one channel per ring
and azimuthal bin. Arguments:

- `max_gb`: GPU memory budget in GB for the batches of orientations that are
  processed together.
- `normalized`: if True, the pole-figure matrix is not scaled by the
  reflection intensities.
- `pf_mode`: `"auto"` (default), `"sparse"` or `"dense"`.
  - The sparse mode evaluates the pole-figure matrix once and stores it in CSR
    format.
  - The dense mode applies it with batched GEMMs, re-evaluating it batch by
    batch when all orientations do not fit in one batch.
  - `"auto"` chooses sparse unless the fill fraction exceeds `sparse_max_fill`
    (default 0.1) or the matrix exceeds `sparse_max_gb`.
- `projector`: `"native"` (default, `ParallelRadon`) or `"gratopy"`. They
  agree to float32 rounding, and the native projector is faster for many
  orientations.
- `verbose`: prints a summary of the GPU buffers.

`support_mask()` returns the `(Ny, Nx)` field-of-view mask used by the solvers.

Array indices are 64-bit, so the coefficient and data arrays may exceed 2³¹
elements. A single orientation batch must stay below 2³¹ elements, and the
operator raises an error that asks for a lower `max_gb` otherwise.

### `MultiPhaseForwardOperator(cfg, materials, grids, two_thetas, max_gb, ...)`

Several materials, each with its own orientation grid. The pole-figure values
of every reflection are convolved with a Gaussian of width `cfg["peak_width"]`
(degrees; change it with `set_peak_width`) onto the detector 2θ bins
`two_thetas` (degrees). The coefficient array has shape `(K_sum, Ny, Nx)`,
with the coefficients of the materials one after the other.

### `BulkTextureForwardOperator(cfg, material, grid, max_gb, ...)`

The pole-figure transform without the tomographic part, `b = PF @ x`, for
bulk texture measurements.

### `MatrixTomographicOperator(B, angles, N_Omega, My, Nx, Ny, K, N_seg, ...)`

Computes `b = B @ P(x)`, where P is the parallel-beam projection and `B` is a
user-supplied matrix of shape `(N_Omega, K, N_seg)`. It also has an
`estimate_L_power` method.

### `BraggEdgeTomographicOperator(material, grid, beam_angles, lam, ...)`

A `MatrixTomographicOperator` for time-of-flight neutron Bragg-edge imaging.
It builds `B` from a material with neutron sites, the orientation grid, the
beam angles (degrees) and the wavelength grid `lam` (Å). The instrument pulse
shape is set by `pulse_tail_fn`, e.g. `diffractom.utils.instrument.raden_pulse_tail`.

### `estimate_L_power(op, niter=20, seed=0, eps=1e-30, verbose=1)`

Estimates the Lipschitz constant ‖AᵀA‖ with a power iteration. The solvers
need an upper bound, so pass a margin, e.g. `L=1.1 * estimate_L_power(op)`.

## `optimization`

| Module | Contents |
|--------|----------|
| `fista_huber.py` | `FISTAHuber`: FISTA with a Huber data term. |
| `fista_l2.py` | `FISTAL2`: FISTA with a least-squares data term. |
| `prox.py` | Non-negativity and L1 proximal operators, support projection. |
| `prox_tv.py` | Non-negative total-variation prox (Chambolle), applied per orientation channel. |
| `launch.py` | Launch sizes for the 64-bit-indexed element-wise kernels. |

### `FISTAHuber` and `FISTAL2`

```python
solver = FISTAHuber(op, prox_kind="nonneg", lam=0.0, L=1.1 * L_est, huber_delta=30)
x = solver.run(x, b, niter=100, weights=w, verbose=1, diagnostics_interval=10)
```

The constructor arguments are:

- `prox_kind`: `"nonneg"`, `"l1"`, `"nonneg_l1"` or `"nonneg_tv"`.
- `lam`: the regularisation weight of the L1 or TV term.
- `L` or `tau`: give `L` (the step is 1/L) or the step `tau` directly.
- `tv_niter`: the number of inner iterations of the TV prox.
- `huber_delta`: the threshold between the quadratic and linear parts of the
  Huber loss. It applies to `FISTAHuber` only and is in the units of the data.
- `support`: the support constraint.
  - `"fov"` (the default) restricts the reconstruction to the operator's
    field-of-view disk, i.e. it assumes that the sample stays in the beam
    during the scan.
  - `None` disables the constraint.
  - An `(Ny, Nx)` boolean array gives a custom support.
- `fused` (default `True`): fuse the gradient step, the prox and the momentum
  update into the adjoint, one orientation batch at a time. Each batch of the
  gradient is consumed as soon as it is computed, so the solver keeps two
  coefficient-sized arrays (`x` and `y`) instead of four (`x`, `y`, `x_old`
  and the gradient), with identical iterates. It applies to the element-wise
  proxes (`"nonneg"`, `"l1"`, `"nonneg_l1"`) with an operator that provides
  `adjoint_batches_cl` (`SinglePhaseForwardOperator`); otherwise the unfused
  update runs.

`run(x, b, niter, ...)` starts from the current values of `x` and returns
the solution. A NumPy `x` streams the coefficients from host memory (below)
and is updated in place if it is C-contiguous float32 (otherwise a converted
copy is updated and returned); a pyopencl `x` keeps everything on the GPU and
is updated in place. `b` may be a NumPy array (uploaded for the run) or a
pyopencl array.

- `weights` (`FISTAHuber` only) holds one weight per segment, i.e. per
  (eta bin, ring): `N_eta * N_rings` elements, e.g. shaped `(N_eta, N_rings)`,
  a NumPy or a C-contiguous float32 pyopencl array. The same weights apply to every
  rotation and translation, so the array is small. Zero weights exclude
  segments, e.g. the eta bins along the rotation axis or gaps between
  detector modules. The residual is computed in place in the prediction
  buffer, so the solver holds two data-sized arrays: `b` and that buffer.
- `verbose` and `diagnostics_interval` control the progress output.
- After the run, `solver.iter_stats` holds one dict per iteration and
  `solver.final_stats` holds a summary.

#### Coefficients in host memory (streaming)

If `x` is a NumPy array (shape `(K, Ny, Nx)`, float32) instead
of a GPU array, `FISTAHuber.run` keeps the coefficient arrays (`x` and `y`) in
host memory and streams them through the GPU one orientation batch at a time.
The GPU then holds the data, the prediction/residual and a few batch-sized
staging buffers, independent of K; host memory holds `x` (updated in place)
and one more array of its size. Transfers overlap the computation (one upload
and one download thread, pinned staging buffers, multi-threaded host copies).
It needs the fused update (an element-wise prox). With large grids the cost is
small: on a V100 with a 400 x 400 grid, 360 rotations, 360 eta bins and 14
rings, an iteration costs 0.84 ms per orientation GPU-resident and 0.83-0.86 ms
per orientation streamed, for K = 3000 up to K = 20000 (2 x 12.8 GB of
coefficients). On small grids the transfers are relatively more expensive
(+12 % on 99 x 99).

```python
op = SinglePhaseForwardOperator(cfg, mat, grid, max_gb=1.0, normalized=True,
                                reserve_coefficient_arrays=0)  # no coefficient arrays on the GPU
L = 1.1 * estimate_L_power_streamed(op, niter=6)
x = np.zeros(op.coeff_shape, np.float32)
x = FISTAHuber(op, prox_kind="nonneg", L=L, huber_delta=100).run(x, b, niter=200, weights=w)
```

`reserve_coefficient_arrays` (default 3) is the number of coefficient-sized
arrays the operator's default sparse-PF budget leaves room for on the GPU;
pass 0 when streaming, so that the sparse PF matrix gets the memory.
`estimate_L_power_streamed(op, niter, seed)` is `estimate_L_power` with its
two coefficient-sized vectors in host memory. `FISTAL2` streams the same way.

## `utils`

| Module | Contents |
|--------|----------|
| `grid.py` | `Grid`, a hierarchical orientation grid of Gaussian kernels. |
| `support.py` | `fov_support_mask(Nx, Ny, n_detectors, angles, ...)`. |
| `reinterpolation/` | Interpolation of coefficients between orientation grids (SO(3)) and of values on the sphere (S²). |
| `gpu_live_tracker.py` | `GPUMemoryLogger`, which polls GPU memory through nvidia-smi. |
| `instrument.py` | `raden_pulse_tail`, the pulse-tail model of the RADEN beamline (J-PARC). |

### `Grid`

Every node holds a rotation and a kernel width sigma (radians). The active
leaf nodes are the basis functions of the operators.

```python
grid = Grid.from_random_fundamental_zone(50000, "cubic", sigma=np.deg2rad(2.0))
grid.prune_close_orientations(theta_deg=3.0, target=15000)   # thin out to about 15000 nodes

grid = Grid.from_rotation_matrices(R_mats, sigma=np.deg2rad(0.5))  # (N, 3, 3) matrices
```

- **Refinement:** `generate_children(parents, radius, stencil=6)` adds child
  nodes around the chosen parents. `set_sigma_for_level(level, sigma)` changes
  the kernel width of a level.
- **Queries:**
  - `active_leaf_nodes()`, `nodes_at_level(level)` and `active_nodes()` return
    node indices.
  - `rotations_at_level(level)` and `scores_at_level(level)` return the
    rotations and scores of a level.
  - `summary()`, `print_summary()` and `inspect_node(i)` describe the tree.
- **Plotting:** `plot_stereographic(level, direction, symmetry)` needs
  matplotlib.

### Reinterpolation

`build_interpolation_kernelSO3(queue, grid_mats, eval_mats, symmetry_ops, sigma_interp)`
builds a sparse GPU kernel that evaluates coefficients given on `grid_mats` at
the orientations `eval_mats`. `interpolateSO3(out, coeffs, kernel, queue)`
applies it. `build_interpolation_kernelS2` and `interpolateS2` do the same for
values on the sphere, and `project_rotations_to_s2(rotations, direction)` maps
orientations to poles.

### `GPUMemoryLogger`

```python
logger = GPUMemoryLogger(interval=0.2, gpu_id=0)
logger.start()
# ... GPU work ...
logger.stop()
t, mem_mb = logger.as_arrays()
```

## Dependencies

- **numpy** and **scipy**: arrays, rotations and KD-trees.
- **pyopencl**: GPU computation.
- **pyclblast** (CLBlast): batched GEMM.
- **gratopy**: the alternative Radon projector.
- **pyvkfft** (optional, `pip install ".[fft]"`): the Fourier-slice projector.
- **matplotlib** (optional): `Grid.plot_stereographic`.
- **h5py**, **orix**, **notebook** (optional, `pip install ".[examples]"`): the example notebook.

CIF files are read by the package's own parser (`cif_parser.py`); no crystallography
package is needed. `environment.yml` pins the versions that were tested.

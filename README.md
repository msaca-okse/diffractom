# diffractom

diffractom is a GPU-accelerated (OpenCL) Python library for texture tomography of scanning diffraction
data, in particular scanning three-dimensional X-ray diffraction (s3DXRD). In such a scan a pencil beam is
translated across a slice of a polycrystalline sample at every rotation step, and each detector frame is
azimuthally integrated over a set of powder rings. From these ring intensities, indexed by rotation angle ω,
translation y, azimuth η and ring, diffractom reconstructs the orientation distribution function (ODF) of
every pixel in the slice.

Because it works with ring intensities rather than individual diffraction spots, the method does not need
the spots of each voxel to be separable and indexable. It also applies to fine-grained, strongly textured or
deformed samples, where the spots merge into rings.

## How it works

- **Model.** The ODF of each pixel is a non-negative combination of K Gaussian kernels centred on an
  orientation grid. The grid can be uniform over the fundamental zone of the crystal symmetry, or tailored
  to the sample, for example from the orientations found by point-by-point indexing.
- **Forward operator.** A pole-figure transform gives, for every kernel orientation, the intensity it
  diffracts into each (ω, η, ring) bin, from the crystal structure and the scan geometry. A parallel-beam
  Radon transform then sums the pixels along the beam at every translation.
- **Inverse problem.** FISTA solves for non-negative coefficients with a Huber or least-squares data term,
  optionally with ℓ1 or total-variation regularisation and per-segment data weights.
- **Scale.** The pole-figure matrix is stored sparse, or generated on the fly when it does not fit in GPU
  memory. Large grids use a Fourier-slice projector. The coefficients can stay in host memory and be
  streamed through the GPU, so the number of orientations and pixels is not limited by GPU memory.

Single-phase data on integrated rings use `SinglePhaseForwardOperator`. The package also contains operators
for multi-phase data on 2θ bins, for bulk texture without tomography, and for neutron Bragg-edge imaging.
See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

## Requirements

- Python 3.11 or newer.
- A working OpenCL installation: GPU drivers (NVIDIA, AMD or Intel) and an
  OpenCL runtime. On HPC systems this usually means running on a GPU node and
  loading the relevant modules. The GPU should appear in the output of
  `python -c "import pyopencl as cl; print(cl.get_platforms())"`.
- Enough GPU memory for the problem. The forward operators take a `max_gb`
  argument that limits the size of the batches they keep on the GPU.

## Installation

Clone the repository:

```bash
git clone https://github.com/msaca-okse/diffractom.git
cd diffractom
```

Create and activate the conda environment. It installs numpy, scipy, pymatgen,
ase, gratopy, pyopencl, pyvkfft, CLBlast and pyclblast:

```bash
conda env create -f environment.yml
conda activate diffractom
```

Install the package in editable mode:

```bash
pip install -e .
```

The example notebook needs a few more packages:

```bash
conda install -c conda-forge orix notebook
pip install h5py matplotlib
```

Conda and pip are mixed on purpose: CLBlast comes from conda-forge, while
pyopencl, pyclblast and gratopy are installed with pip. If pyopencl finds no
platform, or CLBlast fails to load, check the system's OpenCL drivers first.
When several OpenCL devices are available, set `PYOPENCL_CTX` (for example
`PYOPENCL_CTX=0:1`) to choose one.

## Example

`examples/example_diffractom_pipeline.ipynb` reconstructs a small simulated
s3DXRD data set of an aluminium sample (`examples/example_data.h5`: 180 rotation
steps, 103 translations, 180 azimuthal bins and 5 rings, on a 99 × 99 grid). It
does so twice: with an orientation grid tailored to the sample
(`examples/apriori_grid.npy`) and with a uniform grid over the cubic
fundamental zone. It then plots inverse pole figure maps of the dominant
orientation in each pixel. All parameters are set in the notebook, which runs
from the repository root or from the `examples` folder, in about a minute on a
16 GB V100.

## Basic usage

A reconstruction consists of five steps:

1. Describe the scan geometry in a dictionary.
2. Set up the material, from lattice parameters (`Material.from_lattice_parameters`)
   or from a CIF file (`Material.from_cif`).
3. Build an orientation grid.
4. Construct the forward operator.
5. Solve the inverse problem.

The following runs on the example data from the repository root:

```python
import h5py
import numpy as np
from diffractom import Material, Grid, SinglePhaseForwardOperator, FISTAHuber, estimate_L_power_streamed

with h5py.File("examples/example_data.h5", "r") as f:
    data = f["data_array"][...].squeeze()     # (N_Omega, My, N_eta, N_rings)
    lattice = f["lattice"][...]               # direct lattice vectors (Å)
    hkl_list = f["hkl_list"][...]             # one reflection family per ring
    wavelength = float(f["wavelength"][...])  # Å
N_Omega, My, N_eta, N_rings = data.shape

# Scan geometry; directions in the sample frame at rotation angle 0
cfg = {
    "Nx": 99, "Ny": 99,                 # reconstruction grid (pixel size = translation step)
    "My": My,                           # translations
    "N_Omega": N_Omega,                 # rotation steps, the midpoints of equal steps over angle_range
    "angle_range": [0, 180],            # deg
    "cor_offset": 0,                    # centre-of-rotation offset (translation steps)
    "N_eta": N_eta,                     # azimuthal bins, the midpoints of equal bins over eta_angle_range
    "eta_angle_range": [0, 360],        # deg
    "energy": 12.398 / wavelength,      # keV
    "k_direction_0": [0, 0, 1],         # rotation axis
    "p_direction_0": [1, 0, 0],         # beam
    "detector_direction_origin": [0, -1, 0],       # eta = 0 on the detector
    "detector_direction_positive_90": [0, 0, -1],  # eta = 90 deg on the detector
}

# Material: lattice, wavelength and the diffraction rings in the data
mat = Material.from_lattice_parameters(
    lattice_matrix=lattice, symmetry_group="cubic",
    wavelength_A=wavelength, hkl_list=hkl_list,
)

# Orientation grid: Gaussian kernels of width sigma (radians) at random
# orientations in the fundamental zone, pruned to orientations 3 deg apart
grid = Grid.from_random_fundamental_zone(50000, "cubic", np.deg2rad(2.0))
grid.prune_close_orientations(theta_deg=3.0, target=15000)

# Forward operator; reserve_coefficient_arrays=0: the coefficients stay in host memory
op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.5, normalized=True,
                                reserve_coefficient_arrays=0)

# NumPy arrays in and out: data (N_Omega, My, N_eta * N_rings), coefficients (K, Ny, Nx),
# coefficients[k] being the image of orientation k
b = data.reshape(N_Omega, My, N_eta * N_rings)
x = np.zeros(op.coeff_shape, dtype=np.float32)

# FISTA with a Huber data term; x is updated in place (streamed through the GPU)
L = estimate_L_power_streamed(op, niter=6)
solver = FISTAHuber(op, prox_kind="nonneg", L=1.1 * L, huber_delta=30)
x = solver.run(x, b, niter=100, verbose=1, diagnostics_interval=10)
prediction = op.direct(x)  # (N_Omega, My, N_eta * N_rings)
```

[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) lists every key of `cfg`. Any
key can also be passed as a keyword argument of the operator, which overrides
the dictionary.

The FISTA solvers constrain the reconstruction to the field of view by default
(`support="fov"`: the pixels whose centre projects onto the detector at every
angle, i.e. the disk r <= My/2 - |cor_offset| for a full rotation). This
assumes that the sample stays in the beam during the scan. `support=None`
disables the constraint, and an `(Ny, Nx)` boolean array gives a custom
support.

`FISTAL2` is the same solver with a least-squares data term. Both solvers
accept `prox_kind` values `"nonneg"`, `"l1"`, `"nonneg_l1"` and `"nonneg_tv"`
(with the regularisation weight `lam`). `FISTAHuber.run` also takes optional
`weights`, one per segment (eta bin, ring) and shared by all rotations and
translations, where zero weight excludes the segment. See
[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for the full API.

## License

Apache License 2.0

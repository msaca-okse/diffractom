# diffractom

GPU-accelerated texture tomography reconstruction for diffraction data. Each
pixel's orientation distribution is expanded in a basis of Gaussian kernels
on an orientation grid. A combined pole-figure and Radon forward operator maps
the basis coefficients to azimuthally integrated diffraction rings, and FISTA
solves for non-negative coefficients. All heavy computation runs in OpenCL.

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
ase, gratopy, pyopencl, CLBlast and pyclblast:

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
conda install -c conda-forge orix notebook pyyaml
pip install h5py hdf5plugin matplotlib
```

Conda and pip are mixed on purpose: CLBlast comes from conda-forge, while
pyopencl, pyclblast and gratopy are installed with pip. If pyopencl finds no
platform, or CLBlast fails to load, check the system's OpenCL drivers first.
When several OpenCL devices are available, set `PYOPENCL_CTX` (for example
`PYOPENCL_CTX=0:1`) to choose one.

## Example

`examples/example_diffractom_pipeline.ipynb` reconstructs a small simulated
data set (`examples/example_data.h5`, described by `examples/config.yaml`)
twice: with an orientation grid tailored to the sample
(`examples/apriori_grid.npy`) and with a uniform grid over the cubic
fundamental zone. It then plots inverse pole figure maps of the dominant
orientation in each pixel. The notebook runs from the repository root or from
the `examples` folder.

## Basic usage

A reconstruction consists of five steps:

1. Describe the experiment in a config dictionary (see `examples/config.yaml`).
2. Set up the material, from lattice parameters (`Material.from_lattice_parameters`)
   or from a CIF file (`Material.from_cif`).
3. Build an orientation grid.
4. Construct the forward operator.
5. Solve the inverse problem.

The following runs on the example data from the repository root:

```python
import h5py
import numpy as np
import yaml
from diffractom import Material, Grid, SinglePhaseForwardOperator, FISTAHuber, estimate_L_power_streamed

with open("examples/config.yaml") as f:
    cfg = yaml.safe_load(f)
with h5py.File("examples/example_data.h5", "r") as f:
    data = f["data_array"][...].squeeze()   # (N_Omega, My, N_eta, N_theta)
    lattice = f["lattice"][...]
    hkl_list = f["hkl_list"][...]
    wavelength = float(f["wavelength"][...])  # Å

# Material: lattice, wavelength and the diffraction rings in the data
mat = Material.from_lattice_parameters(
    lattice_matrix=lattice, symmetry_group="cubic",
    wavelength_A=wavelength, hkl_list=hkl_list,
)

# Orientation grid: Gaussian kernels of width sigma (radians) at random
# orientations in the fundamental zone, pruned to about 15000 kernels
grid = Grid.from_random_fundamental_zone(50000, "cubic", np.deg2rad(2.0))
grid.prune_close_orientations(theta_deg=3.0, target=15000)
K = len(grid.nodes_at_level(0))

# Forward operator; reserve_coefficient_arrays=0: the coefficients stay in host memory
op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=grid, max_gb=0.5, normalized=True,
                                reserve_coefficient_arrays=0)

# NumPy arrays in and out: data (N_Omega, My, N_eta * N_rings), coefficients (K, Ny, Nx),
# coefficients[k] being the image of orientation k
b = data.reshape(cfg["N_Omega"], cfg["My"], -1)
x = np.zeros(op.coeff_shape, dtype=np.float32)

# FISTA with a Huber data term; x is updated in place (streamed through the GPU)
L = estimate_L_power_streamed(op, niter=6)
solver = FISTAHuber(op, prox_kind="nonneg", L=1.1 * L, huber_delta=30)
x = solver.run(x, b, niter=100, verbose=1, diagnostics_interval=10)
prediction = op.direct(x)  # (N_Omega, My, N_eta * N_rings)
```

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

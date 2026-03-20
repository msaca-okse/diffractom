## Installation

### 1. Clone the repository

```bash
git clone https://github.com/msaca-okse/diffractom.git
cd diffractom
```

### 2. Create and activate environment

We recommend using Conda to manage dependencies:

```bash
conda env create -f environment.yml
conda activate diffractom
```

### 3. Install the package

Install the library in editable mode:

```bash
pip install -e .
```

---

## Requirements

The environment installs all required dependencies, including:

- numpy  
- pyopencl (OpenCL interface)  
- clblast / pyclblast (GPU linear algebra)  
- pymatgen  
- ase  
- gratopy  

---

## GPU / OpenCL requirement

⚠️ **A working OpenCL installation is required.**

This includes:
- GPU drivers (NVIDIA / AMD / Intel)  
- OpenCL runtime available on your system  

On HPC systems, this typically means running on a GPU node and loading the appropriate modules.

---

## Optional dependencies

The following are not required for core functionality, but may be useful for analysis and visualization:

```bash
conda install -c conda-forge orix notebook
pip install h5py hdf5plugin matplotlib
```

---

## Notes

- Mixing `conda` and `pip` is intentional due to GPU/OpenCL dependencies.  
- If you encounter issues with OpenCL or CLBlast, ensure your system configuration is correct.  


## Basic usage

A minimal reconstruction consists of:

1. Define experimental parameters in .yaml config file
2. Setup a material from .cif
2. Building an orientation grid  
3. Constructing the forward operator  
4. Solving the inverse problem  

```python
import diffractom as dt

# Material
material = dt.Material.from_lattice_parameters(...)

# Orientation grid
grid = dt.build_grid("uniform_fz", sigma_rad)

# Forward operator
op = dt.PFO_SINGLE(cfg, material, grid)

# Solve (FISTA)
solver = dt.FISTAHuberOpenCL(op, lam=..., huber_delta=...)
solver.run(x, data, niter=50)


## License
Apache License 2.0

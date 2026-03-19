## Installation

Create and activate a fresh conda environment, install the required dependencies, build the package, and install it:

```bash
conda create --name diffractom python=3.11
conda activate diffractom
```



```bash
conda install numpy
conda install --channel conda-forge pymatgen
pip install --upgrade ase
```

Install 

pip install gratopy
conda install conda-forge::clblast
conda install pyopencl
pip install --user pyclblast

python -m pip install --upgrade pip build
python -m build
python -m pip install dist/diffractom-0.1.0-py3-none-any.whl

# Optional extras:
conda install notebook
pip install h5py hdf5plugin
conda install -c conda-forge orix
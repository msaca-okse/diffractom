## Installation

Create and activate a fresh conda environment, install the required dependencies, build the package, and install it:

```bash
conda create --name textom python=3.11
conda activate textom

conda install numpy
conda install --channel conda-forge pymatgen
pip install --upgrade ase
pip install gratopy

# Optional (depending on your setup):
conda install conda-forge::clblast
conda install pyopencl
pip install --user pyclblast

python -m pip install --upgrade pip build
python -m build
python -m pip install dist/texture_tomography-0.1.0-py3-none-any.whl

# Optional extras:
conda install notebook
pip install h5py hdf5plugin
conda install -c conda-forge orix
#/usr/bin/sh

git clone -b rise_global_experiment https://github.com/SCECcode/pycsep
cd pycsep
conda install -c conda-forge -y cartopy=0.20.2
pip install numpy==1.21.5
conda env update --file requirements.yml --prune
pip install -e .
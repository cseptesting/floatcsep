
## Install pycsep from experiment branch
git clone -b rise_global_experiment https://github.com/SCECcode/pycsep
cd pycsep
conda env update --file requirements.yml --prune
pip install -e .

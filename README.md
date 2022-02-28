# Global Earthquake Forecasting Experiment

## Table of Contents

* [Installing computational environment](installing-computational-environment)
* [Retrieve models](retrieve-models)
* [Downsampling](downsampling)
* [Run tests](tests)
* [Run preliminary experiment](run)


## Installing computational environment


### From conda-forge

```
conda env create -f environment.yml
conda activate gefe
```

Installs pyCSEP and other dependencies to run the experiment. See `environment.yml` file for details.


## Retrieve Models

### Manually from GFZ Gitlab repository
```
sh get_models_from_git.sh
```

## Downsampling models for development

Run the function `gefe.utils.resample_models()` to get a low size version of the models
```python
from gefe.utils import resample_models
resample_models()
```

## Run tests

Run the script `code/tests.py` (i) Check the catalog query, plots the catalog (ii) Check the model exists and plots (iii) Checks the downsampled forecast version and plots.

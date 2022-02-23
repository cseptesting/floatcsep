# Global Earthquake Forecasting Experiment

## Table of Contents

* [Installing computational environment](installing-computational-environment)

## Installing computational environment

### From conda-forge

```
conda env create -f environment.yml
```

Installs pyCSEP and other dependencies to run the experiment. See `environment.yml` file for details.

### From source

Creates a new conda environment, clone the RISE GEFE branch from pyCSEP and installs the required dependencies. 
```
conda env create -n gefe
sh ./Makefile 

```



## Retrieve Forecasts

See `main.py` for details.

### Manually from GFZ Gitlab repository

Clones the repository, creates the forecast folder structure and extract the data.
```
sh ./import_models.sh
```


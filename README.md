# CSEP Floating Experiments - `fecsep`

## Table of Contents

* [Installing computational environment](installing-computational-environment)
* [Run experiment](run-experiment)


## Installing computational environment

Create a `conda` environment

```
conda env create -n $NAME
conda activate $NAME
```

Install `pyCSEP` and its dependencies from `conda`
```
conda install -c conda-forge pycsep
```

Install feCSEP source using `pip`
```
pip install .
```

## Run experiment

Access an experiment directory (e.g. fecsep/examples) and type
```
fecsep run <config> 
```
Usage:
```
    config (required) : path to config.yml file

```

A runtime directory will be created in the `results` folder with the test date as the name. The results from this 
run of the experiment are contained in `readme.md`.


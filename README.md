# CSEP Floating Experiments
<img src="https://i.postimg.cc/4y1q8BZt/fe-CSEP-Logo-CMYK.png" width="320"> 

**A python application to deploy reproducible and prospective earthquake forecasting experiments**

## Table of Contents

* [Installation](installing-computational-environment)
* [Documentation](https://fecsep.readthedocs.io)
* [Set up an Experiment](installing-computational-environment)
* [Run and explore](run-experiment)



## Installing computational environment

The `fecsep` wraps the core functionality of `pycsep` (https://github.com/sceccode/pycsep). To start, create a `conda` environment and install `pyCSEP` and its dependencies from `conda`

```
conda env create -n $NAME
conda activate $NAME
conda install -c conda-forge pycsep

```

Install the feCSEP source code using `pip`
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

A runtime directory will be created in the `results` folder with the test date as the name. The results from the experiment can be visualized in `results/readme.md`.


# CSEP Floating Experiments - `fecsep`

## Table of Contents

* [Installing computational environment](installing-computational-environment)
* [Run experiment](run-experiment)


## Installing computational environment

```
conda env create -f environment.yml
conda activate fecsep
pip install -e .
```

Installs pyCSEP and other dependencies to run the experiment. See `environment.yml` file for details.

## Run experiment
Access the experiment directory
```
fecsep run <config> 
```
Usage:
```
    config (required) : path to config.yml file

```

A runtime directory will be created in the `results` folder with the test date as the name. The results from this 
run of the experiment are contained in `readme.md`.


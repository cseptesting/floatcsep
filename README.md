# CSEP Floating Experiments

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
From the top-level directory type:  
```
python run_experiment.py <test_date>
```
Usage:
```
python run_experiment <test_date>  
    test_date (required) : format='%Y-%m-%dT%H:%M:%S'
```

A runtime directory will be created in the `results` folder with the test date as the name. The results from this 
run of the experiment are contained in `readme.md`.
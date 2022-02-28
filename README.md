# Global Earthquake Forecasting Experiment

## Table of Contents

* [Installing computational environment](installing-computational-environment)
* [Retrieve models](retrieve-models)
* [Run experiment](run-experiment)


## Installing computational environment


### From conda-forge

```
conda env create -f environment.yml
conda activate gefe
```

Installs pyCSEP and other dependencies to run the experiment. See `environment.yml` file for details.


## Retrieve Models

Models are contained within this repository. No extra steps are needed. Forecasts are from the models
each time the experiment is computed. Since models are time-independent, forecasts are simply scaled to the 
appropriate time-horizon for the forecast.

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
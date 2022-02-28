
Global Earthquake Forecast Experiment (GEFE)
============================================

Contents
========

* [Overview](#overview)
	* [Objectives](#objectives)
* [Forecast Experiment](#forecast-experiment)
	* [Evaluation Data](#evaluation-data)
	* [Competing Forecast Models](#competing-forecast-models)
	* [Quadtree Forecasts](#quadtree-forecasts)
	* [Evaluations](#evaluations)
		* [N-test](#n-test)
		* [CL-Test](#cl-test)
		* [M-test](#m-test)
		* [S-test](#s-test)
		* [T-test](#t-test)

# Overview


On 1 January 2006, the Working Group of the Regional Earthquake Likelihood Models (RELM; Field 2007; Schorlemmer et al. 
2007; Schorlemmer and Gerstenberger, 2007) launched an earthquake forecasting experiment to evaluate earthquake 
predictability in California.The RELM experiment sparked a series of subsequent regional forecasting experiments in a 
variety of tectonic settings and the establishment of four testing centers on four different continents (Zechar et al. 
2010; Michael and Werner, 2018; Schorlemmer et al. 2018).

In addition to the regional experiments, CSEP promotes earthquake predictability research at a global scale (Eberhard et
 al. 2012; Taroni et al. 2014; Michael and Werner, 2018; Schorlemmer et al. 2018).Compared to regional approaches, 
global seismicity models offer great testability due to the relatively frequent occurrence of large events worldwide 
(Bayona et al. 2020). In particular, global M5.8+ earthquake forecasting models can be reliably ranked after only one 
year of prospective testing (Bird et al. 2015). In this regard, Eberhard et al. (2012) took a major step toward 
conducting a global forecast experiment by prospectively testing three earthquake forecasting models for the western 
Pacific region. Based on two years of testing, the authors found that a smoothed seismicity model performs the best, and
 provided useful recommendations for future global experiments. Also based on two years of independent observations, 
Strader et al. (2018) determined that the global hybrid GEAR1 model developed by Bird et al. (2015) significantly 
outperformed both of its individual model components, providing preliminary evidence that the combination of smoothed 
seismicity data and interseismic strain rates is suitable for global earthquake forecasting.


## Objectives

- Describe the predictive skills of posited hypothesis about seismogenesis with earthquakes of M5.95+ independent observations around the globe.
- Identify the methods and geophysical datasets that lead to the highest information gains in global earthquake forecasting.
- Test earthquake forecast models on different grid settings.
- Use Quadtree based grid to represent and evaluate earthquake forecasts.

# Forecast Experiment

## Evaluation Data

- The authoritative evaluation data is the full Global CMT catalog (Ekström et al. 2012). 
- We confine the hypocentral depths of earthquakes in training and testing datasets to a maximum of 70km
  
The observed catalog from 2020-01-01 00:00:00 to 2022-12-31 23:59:59:  
![Observed Catalog](code/results/test_catalog.png)
## Competing Forecast Models


TIme independent global seismicity models are provided.The earthquake forecasts are rate-based, i.e. forecast are 
provided as earthquake rates for each longitude/latitute/magnitude bins with bin size of 0.1. All forecasting models 
compute earthquake rates within unique depth bin, i.e. [0 km, 70km]  
Following forecast models are competing in this experiment
- GEAR1 (Bird et al. 2015)
- WHEEL (Bayona et al. 2021)
- TEAM (Bayona et al. 2021)
- KJSS (Kagan and Jackson (2011))
- SHIFT_GSRM2F (Bird & Kreemer (2015))

## Quadtree Forecasts


 The forecast models are provided for the classical CSEP grid. We want to evaluate all these forecasts for multi-
resolution grids. We proposed Quadtree to generate multi-resolution grids. We provided different data-driven multi-
resolution grids based on earthquake catalog and strain data points. We aggregate all the classical forecasts on 
Quadtree grids. Every forecast on a different Quadtree grid is treated as an independent forecast and evaluated 
independently to study the performance of forecast models for different grids.
## Evaluations

### N-test
  
![N-Test](code/results/quadtree_global_experimentN-Test.png)  
The results of N-test from 2020-01-01 00:00:00 to 2022-12-31 23:59:59  
The models passing the N-test are marked with blue dots, while models failing the test are marked with red.
### CL-Test
  
![CL-Test](code/results/quadtree_global_experimentCL-Test.png)  
The results of CL-test from 2020-01-01 00:00:00 to 2022-12-31 23:59:59  
The test shows overall spatial-magnitude consistency of the forecast model with the observation. The models showing 
lower observed likelihood than the confidence interval of of log-likelihoods values are unable to pass the CL-test.
### M-test
  
![M-Test](code/results/quadtree_global_experimentM-Test.png)  
The results of M-test from 2020-01-01 00:00:00 to 2022-12-31 23:59:59  
The test shows the consistency of magnitude aspect of the forecast with the observation. The models showing lower 
observed likelihood than the confidence interval of of log-likelihoods values are unable to pass the M-test.
### S-test
  
![S-Test](code/results/quadtree_global_experimentS-Test.png)  
The results of S-test from 2020-01-01 00:00:00 to 2022-12-31 23:59:59  
The test shows the consistency of spatial aspect of the forecast with the observation. The models showing lower observed
 likelihood than the confidence interval of of log-likelihoods values are unable to pass the S-test.
### T-test
  
![T-Test](code/results/quadtree_global_experimentT-Test.png)  
The results of comparative T-test from 2020-01-01 00:00:00 to 2022-12-31 23:59:59  
The mean information gain per earthquake as is shownc ircles, and the 95 percent confidence interval with vertical 
lines. The models with information gain higher than zero are more informative than the benchmark model, while models 
with lower information gain are less informative.
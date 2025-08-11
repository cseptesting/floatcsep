---
title: 'floatCSEP: An application to deploy and conduct reproducible and prospective earthquake forecasting'
tags:
  - Python
  - seismology
  - forecasting
  - testing
authors:
  - name: Pablo Iturrieta
    orcid: 0000-0002-4787-1343
    corresponding: true
    affiliation: 1
  - name: William H. Savran
    affiliation: 2
  - name: Marcus Herrmann
    affiliation: 3
  - name: Jos\'{e} A. Bayona
    affiliation: 4
  - name: Matt C. Gerstenberger
    affiliation: 5
  - name: Kenny M. Graham
    affiliation: 5
  - name: Philip J. Maechling
    affiliation: 6
  - name: Warner Marzocchi
    affiliation: 3
  - name: Leila Mizrahi
    affiliation: 7
  - name: Danijel Schorlemmer
    affiliation: "1, 7"
  - name: Francesco Serafini
    affiliation: 4
  - name: Fabio Silva
    affiliation: 6
  - name: Maximilian J. Werner
    affiliation: 4
affiliations:
  - name: GFZ Helmholtz Centre for Geosciences, Potsdam, Germany,
    index: 1
  - name: University of Nevada, Reno, United States
    index: 2
  - name: Università degli Studi di Napoli Federico II, Naples, Italy
    index: 3
  - name: School of Earth Sciences, University of Bristol, Bristol, United Kingdom
    index: 4
  - name: GNS Science - Te P\-{u} Ao, Lower Hutt, New Zealand
    index: 5
  - name: Southern California Earthquake Center, University of Southern California
    index: 6
  - name: Swiss Seismological Service at ETH Zurich, Z\:{u}rich, Switzerland
    index: 7
date: 18 August 2025
bibliography: paper.bib
---

# Summary

floatCSEP is a Python application that standardizes and orchestrates the workflow of earthquake forecasting experiments.
Based on principles established by the Collaboratory for the Study of Earthquake Predictability (CSEP, [https://cseptesting.org](https://cseptesting.org)), it enables reproducible, transparent, and reusable evaluations of earthquake forecasts (both prospective or pseudo-prospective).
floatCSEP builds on the existing pyCSEP toolkit for core evaluation routines and adds the functionality needed to deploy and conduct entire experiments, including catalog handling, forecast generation, evaluation, visualization, and reporting.
Accompanying tutorials illustrate experiment use cases, which users can extend to incorporate new models, alternative evaluation metrics, or different regions and timeframes.
Ultimately, floatCSEP will support new official CSEP experiments, and also encourage and empower independent researchers to validate their own models.

# Background

Earthquake forecasts are probabilistic statements about future earthquake occurrence (Jordan et al., 2011), used for informing building codes, emergency response planning, and risk reduction strategies.
Because earthquake occurrence is driven by complex and highly non-linear processes (e.g., Kagan, 1994; Geller et al., 1997), forecasts should be expressed and evaluated in a probabilistic framework designed to describe their fundamental uncertainties (Kagan and Jackson, 1994).
To assess their reliability, further challenges are the large time scales required to collect sufficient observations for evaluation (especially of large earthquakes) and the multiple subjective biases from any post-hoc adjustments in modeling or evaluation environments (e.g., Schorlemmer and Gerstenberger, 2007a).
To address such challenges, the Collaboratory for the Study of Earthquake Predictability (CSEP) was established to facilitate rigorous, prospective forecasting experiments where all forecasting models, data sources, evaluation metrics, and related software are defined prior to the evaluation time period (e.g., Jordan et al., 2006; Schorlemmer et al., 2018).
CSEP experiments were carried out in so-called Testing Centers, i.e., hardware and software infrastructure designed to ensure (i) controlled access, (ii) reproducible environments for automated forecast generation, and (iii) long-term archiving of input data, metadata and results (Zechar et al., 2010).
With this framework, CSEP has successfully hosted and published experiments across diverse geographic regions, such as California, New Zealand, Italy, Japan, and globally (Bayona et al., 2021, 2022, 2023; Eberhard et al., 2012; Field, 2007; Gerstenberger and Rhoades, 2010; Iturrieta et al., 2024; Nanjo et al., 2011; Strader et al., 2018; Taroni et al., 2014, 2018; Tsuruoka et al., 2012; Werner et al., 2010; Zechar et al., 2013).
These efforts have substantially advanced scientific rigor and established community standards in earthquake forecasting research, thereby contributing to better forecasts and seismic hazard assessments (e.g., Michael & Werner, 2018; Schorlemmer et al., 2018).


# Statement of Need

Despite their achievements, the original CSEP Testing Centers were centralized, rigid, and with data management tightly coupled to local hardware, significantly limiting software reusability, scalability, and broader community engagement (e.g., Savran et al., 2022a, Zechar et al., 2010; Schorlemmer et al., 2018).
As also noted by Mizrahi et al. (2024), there is broad consensus in the earthquake forecasting community that transparency and reproducibility are essential in forecast testing.
However, due to the complexity of Testing Centers, independent researchers often require advanced technical expertise to access, reproduce, and analyze CSEP experiments.
To overcome these limitations, the Python package pyCSEP (Savran et al., 2022a, 2022b; Graham et al., 2024) was developed to provide core forecast evaluation routines, which can be directly integrated into modelers’ workflows.


However, pyCSEP alone lacks key features required to deploy and conduct entire forecasting experiments, such as interfacing with external model source code, automating catalog access, managing data, standardizing workflow execution, and generating summary reports.
This lack highlights the need for comprehensive software that provides Testing Center capabilities while remaining decoupled from specific hosting hardware.
The solution should manage the entire experiment lifecycle—from model integration and initial deployment, to the incremental updating of input data, forecasts, results, and reports as new observations become available.


# Software Overview

The primary objective of floatCSEP is to provide a portable, automated, and reproducible Testing Center environment that can run on any computer with sufficient computational resources.
Experiments are defined through human-readable YAML [yaml.org](yaml.org) configuration files, which are processed through a simple command-line interface to ensure ease of use even for users without extensive computational expertise.
This declarative approach simplifies the experiment setup, standardizes its workflow (Figure 1) and enhances its reproducibility.


######## FIGURE 1

floatCSEP uses pyCSEP as a dependency, incorporating its core functionality—forecast and catalog classes, and evaluation routines—alongside additional Testing Center operations, such as data management and computational containerization.
The application supports multiple forecast formats and accommodates both time-invariant and time-variant experiments.
It handles forecasts produced either by models managed directly by floatCSEP or provided externally through raw files.
Representative use cases are included as tutorials, which users can extend by incorporating new models, adding alternative evaluations, or by replicating in different regions and timeframes.
The software integrates seamlessly with pyCSEP’s existing testing routines, but also provides custom hooks for user-defined tests, visualizations, and reports.


# Example Use

An example configuration file (`config.yml`) for a time-invariant, grid-based experiment in Italy with two models is shown in Figure 2.


# Applications

floatCSEP is designed to support the following applications:

- Deploy and conduct new prospective experiments that incrementally incorporate new data, update forecasts, and provide evaluation results. While the CSEP community plans to use floatCSEP for new (“official”) experiments, we also encourage independent researchers to adopt floatCSEP for prospectively evaluating their models
- Reproduce the results of completed prospective CSEP experiments within a containerized computational environment (e.g., Iturrieta et al., 2024).
- Create new retrospective or pseudo-prospective experiments for their easy reproduction and shareability.
- Plug in new models into a completed or ongoing (float)CSEP experiment. Since CSEP experiments are clearly defined, they can be effectively used as benchmarks for comparing and developing new forecasting models (e.g., Serafini et al., 2025).
- Support continuous evaluation of Operational Earthquake Forecasting systems that provide authoritative, near-real-time forecasts (e.g., Jordan et al., 2011; Mizrahi et al., 2024). However, most systems generate forecasts for overlapping windows (e.g., weekly forecasts updated daily) and evaluating the overall performance of such forecast collections remains an open methodological challenge (e.g., Brehmer et al., 2025).

floatCSEP contributes to a growing CSEP software ecosystem that, together with reproducibility packages (e.g., Allison et al., 2018; Bayona et al., 2022, 2023; Savran et al., 2022a; Graham et al., 2024), open-source forecasting models (e.g., Mizrahi et al., 2023), and long-term open-science repositories, could lay the foundation for building robust, collaborative benchmarks in earthquake forecasting.

# Acknowledgements

The development of this software was supported and funded in part by (i) the European Commission under project Geo-INQUIRE [https://www.geo-inquire.eu/](https://www.geo-inquire.eu/), number 101058518 within the HORIZON-INFRA-2021-SERV-01 call; (ii) the European Union H2020 program, Grant number 821115, Real-time earthquake rIsk reduction for a reSilient Europe (RISE, [http://www.rise-eu.org/home/](http://www.rise-eu.org/home/)); (iii) the Statewide California Earthquake Center (Contribution No. 12726). SCEC is funded by NSF Cooperative Agreement EAR-1600087 & USGS Cooperative Agreement G17AC00047; (iv) the U.S. Geological Survey Earthquake Hazards Program under Grant No. G24AP00059; and (v)  the Leverhulme Trust through its Early Career Fellowship program

# References
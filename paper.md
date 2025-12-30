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
  - name: José A. Bayona
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
  - name: GNS Science | Te Pū Ao, Lower Hutt, New Zealand
    index: 5
  - name: Southern California Earthquake Center, University of Southern California, United States
    index: 6
  - name: Swiss Seismological Service at ETH Zürich, Zürich, Switzerland
    index: 7
date: 18 August 2025
bibliography: paper.bib
---

# Summary

floatCSEP is a Python application that standardizes and orchestrates the workflow of earthquake forecasting experiments.
Based on principles established by the Collaboratory for the Study of Earthquake Predictability (CSEP, [https://cseptesting.org](https://cseptesting.org)), it enables reproducible, transparent, and reusable experiments to evaluate earthquake forecasts.
floatCSEP builds on the existing [pyCSEP](https://github.com/cseptesting/pycsep) toolkit for core evaluation routines and adds the functionality needed to deploy and conduct entire experiments, including catalog handling, forecast generation, evaluation, visualization, and reporting.
Accompanying tutorials illustrate experiment use cases, which users can extend to incorporate new models, alternative evaluation metrics, or different regions and timeframes.
Ultimately, floatCSEP is intended to support new official CSEP experiments, and also encourage and empower independent researchers to validate their own models.

# Background

Earthquake forecasts are probabilistic statements about future earthquake occurrence [@jordan2011operational], used for informing building codes, emergency response planning, and risk reduction strategies.
Because earthquake occurrence is driven by complex and highly non-linear processes [e.g., @geller1997earthquake; @kagan1994observational], forecasts should be expressed and evaluated in a probabilistic framework designed to describe their fundamental uncertainties [@kagan1994long].
To assess their reliability, further challenges are the large time scales required to collect sufficient observations for evaluation (especially of large earthquakes) and the multiple subjective biases from any post-hoc adjustments in modeling or evaluation environments [e.g., @schorlemmer2007relm].
To address such challenges, the Collaboratory for the Study of Earthquake Predictability (CSEP) was established to facilitate rigorous, prospective forecasting experiments where all forecasting models, data sources, evaluation metrics, and related software are defined prior to the evaluation time period [e.g., @jordan2006earthquake; @schorlemmer2018collaboratory].
CSEP experiments were carried out in so-called Testing Centers, i.e., hardware and software infrastructure designed to ensure (i) controlled access, (ii) reproducible environments for automated forecast generation, and (iii) long-term archiving of input data, metadata and results [@zechar2010collaboratory].
With this framework, CSEP has successfully hosted and published experiments across diverse geographic regions, such as California, New Zealand, Italy, Japan, and globally [@bayona2021two; @bayona2022prospective; @bayona2023regionally; @eberhard2012prospective; @field2007overview; @gerstenberger2010new; @iturrieta2024evaluation; @nanjo2011overview; @strader2018prospective; @taroni2014assessing; @taroni2018prospective; @tsuruoka2012csep; @werner2010retrospective; @zechar2013regional].
These efforts have substantially advanced scientific rigor and established community standards in earthquake forecasting research, thereby contributing to better forecasts and seismic hazard assessments [e.g., @michael2018preface; @schorlemmer2018collaboratory].


# Statement of Need

Despite their achievements, the original CSEP Testing Centers were centralized, rigid, and with data management tightly coupled to local hardware, significantly limiting software reusability, scalability, and broader community engagement [e.g., @savran2022pycsepsrl; @zechar2010collaboratory; @schorlemmer2018collaboratory].
As also noted by @mizrahi2024developing, there is broad consensus in the earthquake forecasting community that transparency and reproducibility are essential in forecast testing.
However, due to the complexity of Testing Centers, independent researchers often require advanced technical expertise to access, reproduce, and analyze CSEP experiments.
To overcome these limitations, the Python package pyCSEP [@savran2022pycsepsrl; @savran2022pycsepjoss; @graham2024new] was developed to provide core forecast evaluation routines, which can be directly integrated into modelers’ workflows.


However, pyCSEP alone lacks key features required to deploy and conduct entire forecasting experiments, such as interfacing with external model source code, automating catalog access, managing data, standardizing workflow execution, and generating summary reports.
That is, additional software is required that provides Testing Center capabilities while remaining decoupled from specific hosting hardware.
To meet this need, we developed floatCSEP, which manages the entire experiment lifecycle, from model integration and initial deployment to the incremental updating of input data, forecasts, results, and reports as new observations become available.
It is intended for earthquake forecast model developers, institutions that run CSEP-style forecasting experiments, and the broader statistical seismology community.
To our knowledge, no existing software provides this complete end-to-end testing workflow; commonly used seismology tools instead address only individual steps, such as catalog queries (e.g., [ObsPy](https://github.com/obspy/obspy)) or forecast evaluation (e.g., pyCSEP).

# Software Overview

The primary objective of floatCSEP is to provide a portable, automated, and reproducible Testing Center environment that can run on any computer with sufficient computational resources.
Experiments are defined through human-readable YAML ([yaml.org](yaml.org)) configuration files, which are processed through a simple command-line interface to ensure ease of use even for users without extensive computational expertise.
This declarative approach simplifies the experiment setup, standardizes its workflow (\autoref{fig:workflow}) and enhances its reproducibility.

![Workflow diagram of floatCSEP for a time-dependent experiment, which roughly consists of: 1) Defining time-space-magnitude ranges and discretizations for forecast generation and evaluation; 2) Querying and filtering earthquake catalogs (both for model input and evaluation); 3) Building the source code of external models, configuring its parameters and input data; 5) Generating forecasts by running each model source code in a containerized environment; 6) Performing forecast evaluations and comparisons using pyCSEP’s or user-implemented testing metrics; and 7) Producing reports including test results and visual representations. \label{fig:workflow}](figures/fig1_workflow_diagram.png){width="380pt"}

floatCSEP uses pyCSEP as a dependency, incorporating its core functionality (forecast and catalog classes, and evaluation routines) alongside additional Testing Center operations, such as data management and computational containerization [by using [Docker](www.docker.com), @merkel2014docker] .
The application supports multiple forecast formats and accommodates both time-invariant and time-variant experiments.
It handles forecasts produced either by models managed directly by floatCSEP or provided externally through raw files.
Representative use cases are included as tutorials, which users can extend by incorporating new models, adding alternative evaluations, or by replicating in different regions and timeframes.
The software integrates seamlessly with pyCSEP’s existing testing routines, but also provides custom hooks for user-defined tests, visualizations, and reports.


# Example Use

An example configuration file (`config.yml`) for a time-invariant, grid-based experiment in Italy with two models is shown in \autoref{fig:example}.

![Simplest example of a configuration file for a time-invariant, grid-based experiment with two models. The `run` command executes an experiment end-to-end. The `stage` command accesses and builds the models' source code and prepares input/testing catalogs. The `reproduce` command re-executes an experiment and compares it with existing results using statistical and computational metrics. `plot` executes the post-process and visualization of the experiment; The `update` command generates and tests all forecasts missing since the last execution up to today. \label{fig:example}](figures/fig2_example_config.png){width="350pt"}

# Applications

floatCSEP is designed to support the following applications:

- Deploy and conduct new prospective experiments that incrementally incorporate new data, update forecasts, and provide evaluation results. While the CSEP community plans to use floatCSEP for new (official) experiments, we also encourage independent researchers to adopt floatCSEP for prospectively evaluating their models
- Reproduce the results of completed prospective CSEP experiments within a containerized computational environment [e.g., @iturrieta2024evaluation].
- Create new retrospective or pseudo-prospective experiments for their easy reproduction and shareability.
- Plug in new models into a completed or ongoing (float)CSEP experiment. Since CSEP experiments are clearly defined, they can be effectively used as benchmarks for comparing and developing new forecasting models [e.g., @serafini2025benchmark]
- Support continuous evaluation of Operational Earthquake Forecasting systems that provide authoritative, near-real-time forecasts [e.g., @jordan2011operational; @mizrahi2024developing]. However, most systems generate forecasts for overlapping windows (e.g., weekly forecasts updated daily) and evaluating the overall performance of such forecast collections remains an open methodological challenge [e.g., @brehmer2025enhancing].

floatCSEP contributes to a growing CSEP software ecosystem that, together with reproducibility packages [e.g., @allison2018reproducibility; @bayona2022prospective; @bayona2023regionally;  @savran2022pycsepsrl; @graham2024new; @iturrieta2024evaluation], open-source forecasting models [e.g., @mizrahi2023etas] , and long-term open-science repositories, could lay the foundation for building robust, collaborative benchmarks in earthquake forecasting.

# Acknowledgements

Details of author contributions can be found in [```CREDITS.md```](https://github.com/cseptesting/floatcsep/blob/c9b831f804c14129c6c20cc5618234a456dcfbc7/CREDITS.md) in the code repository.
The development of this software was supported and funded in part by (i) the European Commission under project Geo-INQUIRE [https://www.geo-inquire.eu/](https://www.geo-inquire.eu/), number 101058518 within the HORIZON-INFRA-2021-SERV-01 call; (ii) the European Union H2020 program, Grant number 821115, Real-time earthquake rIsk reduction for a reSilient Europe (RISE, [http://www.rise-eu.org/home/](http://www.rise-eu.org/home/)); (iii) the Statewide California Earthquake Center (Contribution No. 14983). SCEC is funded by NSF Cooperative Agreement EAR-2225216 and USGS Cooperative Agreement G24AC00072-00; (iv) the U.S. Geological Survey Earthquake Hazards Program under Grant Nos. G24AP00059 and G25AP00379; and (v) the Leverhulme Trust through its Early Career Fellowship program.

# References
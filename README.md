# CSEP Floating Experiments

<img src="https://i.postimg.cc/6p5krRnB/float-CSEP-Logo-CMYK.png" width="320"> 

**An application to deploy reproducible and prospective experiments of earthquake forecasting**

<p left>

<a href='https://floatcsep.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/floatcsep/badge/?version=latest' alt='Documentation Status' />
</a>
<a href='https://github.com/cseptesting/floatcsep/actions/workflows/build-test.yml'>
    <img src='https://github.com/cseptesting/floatcsep/actions/workflows/build-test.yml/badge.svg' alt='Documentation Status' />
</a>
<img alt="PyPI" src="https://img.shields.io/pypi/v/floatcsep">

<a href="https://codecov.io/gh/cseptesting/floatcsep" > 
 <img src="https://codecov.io/gh/cseptesting/floatcsep/branch/main/graph/badge.svg?token=LI4RSDOKA1"/> 
 </a>
<a href="https://doi.org/10.5281/zenodo.7953817"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7953817.svg" alt="DOI"></a>
</p>

* Set up a testing **experiment** for your earthquake forecasts using authoritative data sources and benchmarks.
* **Encapsulate** the complete experiment's definition and rules in a couple of lines.
* **Reproduce**, **reuse**, and **share** forecasting experiments from you, other researchers and institutions.

# Table of Contents

* [Installation](#installing-floatcsep)
* [Documentation](https://floatcsep.readthedocs.io)
* [Run and explore](#run-an-experiment)
* [Useful Links](#important-links)
* [Roadmap/Issues](#roadmap-and-known-issues)
* [Contributing](#contributing)
* [License](#license)


# Installing floatCSEP

The core of `floatCSEP` is built around the `pyCSEP` package (https://github.com/sceccode/pycsep), which itself contains the core dependencies. 

The simplest way to install `floatCSEP`, is by creating a `conda` environment (https://conda.io - checkout Anaconda or Miniconda) and install `pyCSEP` from `conda-forge`

```
conda env create -n $NAME
conda activate $NAME
conda install -c conda-forge pycsep
```

Clone and install the floatCSEP source code using `pip`
```
git clone https://github.com/cseptesting/floatcsep
cd floatcsep
pip install .
```

Please read the [Installation](https://floatcsep.readthedocs.io/en/latest/intro/installation.html) documentation for detailed instructions and additional installation methods.

# Run an Experiment

Using the command terminal, navigate to an example experiment in `floatcsep/examples/` and type
```
floatcsep run config.yml
```
A runtime directory will be created in a `results` folder. The experiment results can be visualized in `results/report.md`. **Check out the experiment, models and tests definition in the examples**! 

# Important Links

* [Documentation](https://floatcsep.readthedocs.io/en/latest/)
* [CSEP Website](https://cseptesting.org)
* `pyCSEP` [Github](https://github.com/sceccode/pycsep)
* `pyCSEP` [Documentation](https://docs.cseptesting.org/)

# Roadmap and Known Issues

* Add functionality to compare original results and reproduced results
* Add registry to filetrees (e.g. hash/byte count) for a proper experiment re-instantiation
* Add report customization
* Fix the hooks properly (user code) to be inserted into plotting/reporting functionalities.
* Add multiple logging/levels
* Create tool to check a TD model's interface with ``fecsep``
* Define a dependency strategy to ensure experiments' reproducibility.
* Publish in `conda`

# Contributing

We encourage all types of contributions, from reporting bugs, suggesting enhancements, adding new features and more. Please refer to the [Contribution Guidelines](https://github.com/cseptesting/floatcsep/blob/main/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/cseptesting/floatcsep/blob/main/CODE_OF_CONDUCT.md) for more information

# License

The `floatCSEP` (as well as `pyCSEP`) software is distributed under the BSD 3-Clause open-source license. Please see the [license file](https://github.com/cseptesting/floatcsep/blob/main/LICENSE) for more information.
# CSEP Floating Experiments

<img src="https://i.postimg.cc/4y1q8BZt/fe-CSEP-Logo-CMYK.png" width="320"> 

**An application to deploy reproducible and prospective experiments of earthquake forecasting**

<p left>

<a href='https://fecsep.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/fecsep/badge/?version=latest' alt='Documentation Status' />
</a>
<a href='https://github.com/cseptesting/fecsep/actions/workflows/build-test.yml'>
    <img src='https://github.com/cseptesting/fecsep/actions/workflows/build-test.yml/badge.svg' alt='Documentation Status' />
</a>
</p>

* Set up a testing **experiment** for your earthquake forecasts using authoritative data sources and benchmarks.
* **Encapsulate** the complete experiment's definition and rules in a couple of lines.
* **Reproduce**, **reuse**, and **share** forecasting experiments from you, other researchers and institutions.

# Table of Contents

* [Installation](#installing-fecsep)
* [Documentation](https://fecsep.readthedocs.io)
* [Run and explore](#run-an-experiment)
* [Useful Links](#important-links)
* [Roadmap/Issues](#roadmap-and-known-issues)
* [Contributing](#contributing)
* [License](#license)


# Installing feCSEP

The core of `feCSEP` is built around the `pyCSEP` package (https://github.com/sceccode/pycsep), which itself contains the core dependencies. 

The simplest way to install `feCSEP`, is by creating a `conda` environment (https://conda.io - checkout Anaconda or Miniconda) and install `pyCSEP` from `conda-forge`

```
conda env create -n $NAME
conda activate $NAME
conda install -c conda-forge pycsep
```

Clone and install the feCSEP source code using `pip`
```
git clone https://github.com/cseptesting/fecsep
cd fecsep
pip install .
```

Please read the [Installation](https://fecsep.readthedocs.io/en/latest/intro/installation.html) documentation for detailed instructions and additional installation methods.

# Run an Experiment

Using the command terminal, navigate to an example experiment in `fecsep/examples/` and type
```
fecsep run config.yml
```
A runtime directory will be created in a `results` folder. The experiment results can be visualized in `results/report.md`. **Check out the experiment, models and tests definition in the examples**! 

# Important Links

* [Documentation](https://fecsep.readthedocs.io/en/latest/)
* [CSEP Website](https://cseptesting.org)
* `pyCSEP` [Github](https://github.com/sceccode/pycsep)
* `pyCSEP` [Documentation](https://docs.cseptesting.org/)

# Roadmap and Known Issues

* Add functionality to compare original results and reproduced results
* Add registry to filetrees (e.g. hash/byte count) for a proper experiment re-instantiation
* Add interface to time-dependent models
* Add report customization
* Allow hooks (user code) to be inserted into plotting/reporting functionalities.
* Add multiple logging/levels
* Define a strategy to handle dependences to ensure experiments' reproducibility.
* Publish in `pypi`/`conda`

# Contributing

We encourage all types of contributions, from reporting bugs, suggesting enhancements, adding new features and more. Please refer to the [Contribution Guidelines](https://github.com/cseptesting/fecsep/blob/main/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/cseptesting/fecsep/blob/main/CODE_OF_CONDUCT.md) for more information

# License

The `feCSEP` (as well as `pyCSEP`) software is distributed under the BSD 3-Clause open-source license. Please see the [license file](https://github.com/cseptesting/fecsep/blob/main/LICENSE) for more information.
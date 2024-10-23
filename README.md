# CSEP Floating Experiments

<img src="https://i.postimg.cc/6p5krRnB/float-CSEP-Logo-CMYK.png" width="320"> 

**An application to deploy reproducible and prospective experiments of earthquake forecasting**

<p left>

[![Documentation Status](https://readthedocs.org/projects/floatcsep/badge/?version=latest)](https://floatcsep.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/cseptesting/floatcsep/actions/workflows/build-test.yml/badge.svg)](https://github.com/cseptesting/floatcsep/actions/workflows/build-test.yml)
[![PyPI Version](https://img.shields.io/pypi/v/floatcsep)](https://pypi.org/project/floatcsep/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/floatcsep)](https://anaconda.org/conda-forge/floatcsep)
[![Python Versions](https://img.shields.io/pypi/pyversions/floatcsep)](https://pypi.org/project/floatcsep/)
[![Code Coverage](https://codecov.io/gh/cseptesting/floatcsep/branch/main/graph/badge.svg?token=LI4RSDOKA1)](https://codecov.io/gh/cseptesting/floatcsep)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7953816.svg)](https://doi.org/10.5281/zenodo.7953816)


</p>

* Set up a testing **experiment** for your earthquake forecasts using authoritative data sources
  and benchmarks.
* **Encapsulate** the complete experiment's definition and rules in a couple of lines.
* **Reproduce**, **reuse**, and **share** forecasting experiments from you, other researchers
  and institutions.

# Table of Contents

* [Installation](#installing-floatcsep)
* [Documentation](https://floatcsep.readthedocs.io)
* [Run and explore](#run-an-experiment)
* [Useful Links](#important-links)
* [Roadmap/Issues](#roadmap-and-known-issues)
* [Contributing](#contributing)
* [License](#license)

# Installing floatCSEP

The core of `floatCSEP` is built around the `pyCSEP`
package (https://github.com/sceccode/pycsep), which itself contains the core dependencies.

The simplest way to install `floatCSEP`, is by creating a `conda`
environment (https://conda.io - checkout Anaconda or Miniconda) and install `pyCSEP`
from `conda-forge`

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

Please read
the [Installation](https://floatcsep.readthedocs.io/en/latest/intro/installation.html)
documentation for detailed instructions and additional installation methods.

# Run an Experiment

Using the command terminal, navigate to an example experiment in ``floatcsep/tutorials/`` and
type

```
floatcsep run config.yml
```

A runtime directory will be created in a `results` folder. The experiment results can be
visualized in `results/report.md`. **Check out the experiment, models and tests definition in
the tutorials**!

# Important Links

* [Documentation](https://floatcsep.readthedocs.io/en/latest/)
* [CSEP Website](https://cseptesting.org)
* `pyCSEP` [Github](https://github.com/sceccode/pycsep)
* `pyCSEP` [Documentation](https://docs.cseptesting.org/)

# Roadmap and Known Issues

* Create tool to check a TD model's interface with ``floatcsep``
* Define a dependency strategy to ensure experiments' reproducibility (e.g., storing docker image).
* Implement spatial database and HDF5 experiment storage feature
* Set up task parallelization
* Document the process of an experiment deployment

# Contributing

We encourage all types of contributions, from reporting bugs, suggesting enhancements, adding
new features and more. Please refer to
the [Contribution Guidelines](https://github.com/cseptesting/floatcsep/blob/main/CONTRIBUTING.md)
and the [Code of Conduct](https://github.com/cseptesting/floatcsep/blob/main/CODE_OF_CONDUCT.md)
for more information

# License

The `floatCSEP` (as well as `pyCSEP`) software is distributed under the BSD 3-Clause open-source
license. Please see
the [license file](https://github.com/cseptesting/floatcsep/blob/main/LICENSE) for more
information.

## Support

<div style="max-width: 100px; margin: auto;">

| <img src="https://i.postimg.cc/tC1LdjYf/scec.png" width="150"/>                                                                                                                                                                                           | <img src="https://i.postimg.cc/2S7ThcFM/eu-flag.jpg" width="150"/>                                                                                                                                                                                                                                                               |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="text-align: left;">This research was supported by the <a href="https://www.scec.org/">Statewide California Earthquake Center</a>. SCEC is funded by NSF Cooperative Agreement EAR-2225216 and USGS Cooperative Agreement G24AC00072-00.</div> | <div style="text-align: left;">The work in this repository has received funding from the European Union’s Horizon research and innovation programme under grant agreements No.s 101058518 and 821115 of the projects <a href="https://www.geo-inquire.eu/">GeoInquire</a> and <a href="https://www.rise-eu.org/">RISE</a>.</div> |

</div>

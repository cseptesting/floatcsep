===============================
floatCSEP: Floating Experiments
===============================

*Testing earthquake forecasts made simple.*

.. image:: https://img.shields.io/badge/GitHub-Repository-blue?logo=github
   :target: https://github.com/cseptesting/floatcsep
   :alt: GitHub Repository

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7953816.svg
   :target: https://doi.org/10.5281/zenodo.7953816
   :alt: Zenodo

.. image:: https://img.shields.io/github/license/cseptesting/floatcsep.svg
   :target: https://github.com/cseptesting/floatcsep/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/pypi/v/floatcsep.svg
   :target: https://pypi.org/project/floatcsep/
   :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/floatcsep.svg
   :target: https://anaconda.org/conda-forge/floatcsep
   :alt: Conda Version



.. |start| image:: https://img.icons8.com/office/40/rocket.png
    :target: intro/installation.html
    :height: 48px

.. |learn1| image:: https://img.icons8.com/nolan/64/literature.png
   :target: https://docs.cseptesting.org/getting_started/core_concepts.html
   :height: 48px

.. |learn2| image:: https://i.postimg.cc/wMTtqBGB/icons8-literature-64.png
   :target: https://docs.cseptesting.org/getting_started/theory.html
   :height: 48px

.. |experiment| image:: https://img.icons8.com/pulsar-color/48/science-application.png
   :height: 48px

.. |api| image:: https://img.icons8.com/nolan/64/code--v2.png
   :target: reference/api_reference.html
   :height: 48px

.. |tutorials| image:: https://img.icons8.com/nolan/64/checklist.png
   :target: examples/case_a.html
   :height: 48px


Quickstart
----------

+--------------------------------------------------+-------------------------------------+
| |start| **Get Started**                          | |tutorials| **Tutorials**           |
|                                                  |                                     |
| |learn1| **Forecasting Concepts**                | - :ref:`example_a`                  |
|                                                  | - :ref:`example_b`                  |
| |learn2| **Testing Theory**                      | - :ref:`example_c`                  |
|                                                  | - :ref:`example_d`                  |
| |experiment| **Floating Experiments**            | - :ref:`example_e`                  |
|                                                  | - :ref:`example_f`                  |
| |api| **API Reference**                          | - :ref:`example_g`                  |
|                                                  | - :ref:`example_h`                  |
+--------------------------------------------------+-------------------------------------+

What is floatCSEP
-----------------

The `Collaboratory for the Study of Earthquake Predictability <https://cseptesting.org>`_ (CSEP) has organized Earthquake Forecast Testing Experiments during the last decades and is now consolidating its research into open-software initiatives.

**floatCSEP** is an easy-to-use software application that contains the workflow to deploy Earthquake Forecasting Experiments. It is based on the code python library **pyCSEP** (`Github <https://github.com/sceccode/pycsep>`_), which itself contains the core routines to test earthquake forecasts.

Goals
-----

    * Set up a testing experiment for your forecasts using authoritative data sources/benchmarks.
    * Encapsulate the complete experiment's definition and rules in a couple of lines.
    * Produce human-readable results and figures.
    * Reproduce, reuse, and share forecasting experiments.

Running
-------

Start using **floatCSEP** by `installing <intro/installation.html>`_ the latest version and running the ``examples`` with simply:

.. code-block::

   $ floatcsep run config.yml

Useful Links
------------

+---------------------------------------------------------+-----------------------------------------+
| .. image:: https://img.icons8.com/nolan/64/github.png   | **GitHub Repository**                   |
|   :height: 48px                                         |                                         |
|   :target: https://github.com/cseptesting/floatcsep     |                                         |
+---------------------------------------------------------+-----------------------------------------+
| .. image:: https://i.postimg.cc/HW2Pssx1/logo-csep.png  | **CSEP Website**                        |
|   :height: 48px                                         |                                         |
|   :target: https://cseptesting.org                      |                                         |
+---------------------------------------------------------+-----------------------------------------+
| .. image:: https://img.icons8.com/nolan/64/github.png   | **pyCSEP GitHub**                       |
|   :height: 48px                                         |                                         |
|   :target: https://github.com/sceccode/pycsep           |                                         |
+---------------------------------------------------------+-----------------------------------------+
| .. image:: https://img.icons8.com/nolan/64/europe.png   | **European Testing Center**             |
|   :height: 48px                                         |                                         |
|   :target: http://eqstats.efehr.org                     |                                         |
+---------------------------------------------------------+-----------------------------------------+




Collaborators
-------------

    * Pablo Iturrieta, GFZ Potsdam (pciturri@gfz-potsdam.de)
    * William Savran, University of Nevada, Reno
    * Jose Bayona, University of Bristol
    * Francesco Serafini, University of Edinburgh
    * Khawaja Asim, GFZ Potsdam
    * Fabio Silva, Southern California Earthquake Center
    * Marcus Hermann, University of Naples ‘Frederico II’
    * Max Werner, University of Bristol
    * Danijel Schorlemmner, GFZ Potsdam
    * Philip Maechling, Southern California Earthquake Center




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Get Started

   intro/installation.rst
   intro/forecasting_experiments.rst
   intro/floating_experiments.rst


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorial Experiments

   examples/case_a.rst
   examples/case_b.rst
   examples/case_c.rst
   examples/case_d.rst
   examples/case_e.rst
   examples/case_f.rst
   examples/case_g.rst
   examples/case_h.rst

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Help & Reference

   reference/api_reference

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Defining an Experiment

   guide/config.rst
   guide/time_config.rst
   guide/region_config.rst
   guide/model_config.rst
   guide/tests_config.rst
   guide/postprocess_config.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Deploying an Experiment

   deployment/intro.rst





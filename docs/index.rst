===============================
floatCSEP: Floating Experiments
===============================
*Earthquake forecasting experiments made simple.*

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
   :target: intro/concepts.html
   :height: 48px

.. |api| image:: https://img.icons8.com/nolan/64/code--v2.png
   :target: reference/api_reference.html
   :height: 48px

.. |tutorials| image:: https://img.icons8.com/nolan/64/checklist.png
   :target: tutorials/case_a.html
   :height: 48px


Quickstart
----------

+--------------------------------------------------+-------------------------------------+
| |start| **Get Started**                          | |tutorials| **Tutorials**           |
|                                                  |                                     |
| |learn1| **Forecasting Concepts**                | - :ref:`case_a`                     |
|                                                  | - :ref:`case_b`                     |
| |learn2| **Testing Theory**                      | - :ref:`case_c`                     |
|                                                  | - :ref:`case_d`                     |
| |experiment| **Experiment Concepts**             | - :ref:`case_e`                     |
|                                                  | - :ref:`case_f`                     |
| |api| **API Reference**                          | - :ref:`case_g`                     |
|                                                  | - :ref:`case_h`                     |
+--------------------------------------------------+-------------------------------------+

What is floatCSEP
-----------------

The `Collaboratory for the Study of Earthquake Predictability <https://cseptesting.org>`_ (CSEP) has organized Earthquake Forecast Testing Experiments during the last decades and is now consolidating its research into open-software initiatives.

**floatCSEP** is an easy-to-use software application that contains the workflow to deploy Earthquake Forecasting Experiments. It is based on the code python library **pyCSEP** (`Github <https://github.com/sceccode/pycsep>`_), which itself contains the core routines to test earthquake forecasts.

Goals
-----

* Test your forecasts with simple commands.
* Set up a testing experiment for your forecasts using authoritative data sources/benchmarks.
* Encapsulate the complete experiment's definition and rules in a couple of lines.
* Reproduce, reuse, and share forecasting experiments.

Running
-------

Start using **floatCSEP** by `installing <intro/installation.html>`_ the latest version and running the ``tutorials`` with simply:

.. code-block:: console

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
| .. image:: https://img.icons8.com/nolan/64/github.png   | **hazard2csep GitHub**                  |
|   :height: 48px                                         |                                         |
|   :target: https://github.com/cseptesting/hazard2csep   |                                         |
+---------------------------------------------------------+-----------------------------------------+
| .. image:: https://img.icons8.com/nolan/64/europe.png   | **European Testing Center**             |
|   :height: 48px                                         |                                         |
|   :target: http://eqstats.efehr.org                     |                                         |
+---------------------------------------------------------+-----------------------------------------+




Collaborators
-------------

    * Pablo Iturrieta, GFZ Potsdam, Germany (pciturri@gfz-potsdam.de)
    * William Savran, University of Nevada, Reno, USA
    * Jose Bayona, University of Bristol, United Kingdom
    * Francesco Serafini, University of Edinburgh, United Kingdom
    * Kenny Graham, GNS Science, New Zealand
    * Khawaja Asim, GFZ Potsdam, Germany
    * Fabio Silva, Southern California Earthquake Center, USA
    * Marcus Hermann, University of Naples ‘Frederico II’, Italy
    * Max Werner, University of Bristol, United Kingdom
    * Danijel Schorlemmner, GFZ Potsdam, Germany
    * Philip Maechling, Southern California Earthquake Center, USA




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Get Started

   intro/installation.rst
   intro/concepts.rst


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorial Experiments

   tutorials/case_a.rst
   tutorials/case_b.rst
   tutorials/case_c.rst
   tutorials/case_d.rst
   tutorials/case_e.rst
   tutorials/case_f.rst
   tutorials/case_g.rst
   tutorials/case_h.rst


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Defining an Experiment

   guide/experiment_config.rst
   guide/model_config.rst
   guide/evaluation_config.rst
   guide/postprocess_config.rst
   guide/executing_experiment.rst


.. sidebar-links::
   :caption: Help & Reference
   :github:

   reference/api_reference
   Getting Help <https://github.com/cseptesting/floatcsep/issues>
   Contributing <https://github.com/cseptesting/floatcsep/blob/master/CONTRIBUTING.md>
   License <https://github.com/cseptesting/floatcsep/blob/master/LICENSE>

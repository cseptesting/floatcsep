Installation
============

.. important::

    This application uses ``3.9 <= python <= 3.11``


Latest Version
--------------

This option is recommended to learn the software, run the tutorials, and drafting **Testing Experiments**.

1. Using ``conda``
~~~~~~~~~~~~~~~~~~

First, clone the **floatCSEP** source code into a new directory by typing into a terminal:

    .. code-block:: console

        $ git clone https://github.com/cseptesting/floatcsep
        $ cd floatcsep

Then, let ``conda`` automatically install all required dependencies of **floatCSEP** (from its ``environment.yml`` file) into a new environment, and activate it:

    .. code-block:: console

        $ conda env create -n csep_env
        $ conda activate csep_env

.. note::

    For this to work, you need to have ``conda`` installed (see `conda.io <https://conda.io>`_), either by installing the `Anaconda Distribution <https://docs.anaconda.com/anaconda/install/>`_,
    or its more minimal variants `Miniconda <https://docs.anaconda.com/miniconda/>`_ or `Miniforge <https://conda-forge.org/download>`_ (recommended).
    If you install `Miniforge`, we further recommend to use the ``mamba`` command instead of ``conda`` (a faster drop-in replacement).


Lastly, install **floatCSEP** into the new environment using ``pip``:

    .. code-block:: console

        $ pip install .

.. note::

    To *update* **floatCSEP** and its dependencies at a later date, simply execute:

        .. code-block:: console

            $ conda env update --file environment.yml
            $ pip install . -U


2. Using only ``pip``
~~~~~~~~~~~~~~~~~~~~~

To install using the ``pip`` manager only, we require to install the binary dependencies of **pyCSEP** (see `Installing pyCSEP <https://docs.cseptesting.org/getting_started/installing.html>`_). The **floatCSEP** latest version can then be installed as:

    .. code-block:: console

        $ git clone https://github.com/cseptesting/floatcsep
        $ cd floatcsep
        $ python -m venv venv
        $ pip install .


Latest Stable Release
---------------------

This option is recommended for deploying *Floating Testing Experiments* live.

1. From the ``conda-forge`` channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having a ``conda`` manager installed (see **Note** box above), type in a console:

    .. code-block:: console

        $ conda create -n csep_env
        $ conda activate csep_env
        $ conda install -c conda-forge floatcsep


2. From the ``PyPI`` repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having installed the binary dependencies of **pyCSEP** (see `Installing pyCSEP <https://docs.cseptesting.org/getting_started/installing.html>`_}, install **floatCSEP** by:

    .. code-block:: console

        $ python -m venv venv
        $ pip install floatcsep

.. important::
    If you want to run the tutorials from a **floatCSEP** installation obtained through ``conda-forge`` or ``PyPI``, the tutorials can be downloaded to your current directory as:

    .. code-block:: console

        $ latest_version=$(curl --silent "https://api.github.com/repos/cseptesting/floatcsep/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/') && \
        wget "https://github.com/cseptesting/floatcsep/releases/download/$latest_version/tutorials.zip"
        $ unzip tutorials.zip -d ./ && rm tutorials.zip

    Or downloaded manually from the `latest release  <https://github.com/cseptesting/floatcsep/releases>`_.



For Developers
--------------

It is recommended (not obligatory) to use a ``conda`` environment to make sure your contributions do not depend on your system local libraries. For contributing to the **floatCSEP** codebase, please consider `forking the repository <https://docs.github.com/articles/fork-a-repo>`_ and `create pull-requests <https://docs.github.com/articles/creating-a-pull-request>`_ from there.

    .. code-block:: console

        $ conda env create -n csep_dev
        $ conda activate csep_dev
        $ git clone https://github.com/${your_fork}/floatcsep
        $ cd floatcsep
        $ pip install .[dev]

This will install and configure all the unit-testing, linting, and documentation packages.

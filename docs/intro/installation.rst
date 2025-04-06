Installation
============

.. important::

    This application uses ``3.9 <= python <= 3.11``


Latest Version
--------------

Recommended to learn the software, run the tutorials, and drafting **Testing Experiments**.

1. Using ``conda``
~~~~~~~~~~~~~~~~~~

To install **floatCSEP**, first a ``conda`` manager should be installed (https://conda.io). Checkout `Anaconda`, `Miniconda` or `Miniforge` (recommended). Once installed, create an environment with:

    .. code-block:: console

        $ conda create -n csep_env
        $ conda activate csep_env

Then, clone and install the floatCSEP source code using ``pip``

    .. code-block:: console

        $ git clone https://github.com/cseptesting/floatcsep
        $ cd floatcsep
        $ pip install .

.. note::

    Use the ``mamba`` command instead of ``conda`` if `Miniforge` was installed.


2. Using ``pip`` only
~~~~~~~~~~~~~~~~~~~~~

To install using the ``pip`` manager only, we require to install the binary dependencies of **pyCSEP** (see `Installing pyCSEP <https://docs.cseptesting.org/getting_started/installing.html>`_}. The **floatCSEP** latest version can then be installed as:

    .. code-block:: console

        $ git clone https://github.com/cseptesting/floatcsep
        $ cd floatcsep
        $ python -m venv venv
        $ pip install .


Latest Stable Release
---------------------

Recommended for deploying live Floating Testing Experiments

1. From the ``conda-forge`` channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having a ``conda`` manager installed (https://conda.io), type in a console:


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

It is recommended (not obligatory) to use a ``conda`` environment to make sure your contributions do not depend on your system local libraries. For contributions to the **floatCSEP** codebase, please consider using a `fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_ and creating pull-requests from there.

    .. code-block:: console

        $ conda create -n csep_dev
        $ conda activate csep_dev
        $ git clone https://github.com/${your_fork}/floatcsep
        $ cd floatcsep
        $ pip install .[dev]

This will install and configure all the unit-testing, linting and documentation packages.

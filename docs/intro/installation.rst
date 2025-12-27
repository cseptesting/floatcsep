Installation
============

.. important::

    This application uses ``3.9 <= python <= 3.12``


Latest Version
--------------

This option is recommended to learn the software, run the tutorials, and drafting **Testing Experiments**.

.. note::

    We recommend installing with ``conda`` because it bundles native/system dependencies.


.. _conda-install-note:

1. Using ``conda``
~~~~~~~~~~~~~~~~~~

First, clone the **floatCSEP** source code into a new directory by typing into a terminal:

    .. code-block:: console

        $ git clone https://github.com/cseptesting/floatcsep
        $ cd floatcsep

Then, let ``conda`` automatically install all required dependencies of **floatCSEP** (from its ``environment.yml`` file) into a new environment, and activate it:

    .. code-block:: console

        $ conda env create -f environment.yml
        $ conda activate floatcsep

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

            $ git pull
            $ conda env update --file environment.yml
            $ pip install . -U


2. Using only ``pip``
~~~~~~~~~~~~~~~~~~~~~

To install using the ``pip`` manager only, you need the binary dependencies of **pyCSEP**
(see `Installing pyCSEP <https://docs.cseptesting.org/getting_started/installing.html>`_).

.. note::

    Pip-only installs may require native libraries for PDF report generation (WeasyPrint).
    See :ref:`pip-binary-deps`.

The **floatCSEP** latest version can then be installed as:

.. code-block:: console

    $ git clone https://github.com/cseptesting/floatcsep
    $ cd floatcsep
    $ python -m venv venv
    $ pip install .


.. _pip-binary-deps:

Binary dependencies for pip-only installs
-----------------------------------------

Debian/Ubuntu:

    .. code-block:: console

        $ sudo apt-get update
        $ sudo apt-get install -y \
            libglib2.0-0 \
            libpango-1.0-0 \
            libpangoft2-1.0-0 \
            libharfbuzz0b \
            libharfbuzz-subset0

macOS (Homebrew):

    .. code-block:: console

        $ brew install cairo pango gdk-pixbuf libffi


In macOS, if ``import weasyprint`` fails to find libraries (e.g. ``libgobject-2.0-0``),
set this for the current terminal session:

.. code-block:: console

      $ export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:/usr/local/lib:$DYLD_FALLBACK_LIBRARY_PATH"
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:/usr/local/lib:$DYLD_FALLBACK_LIBRARY_PATH"

Latest Stable Release
---------------------

This option is recommended for deploying *Floating Testing Experiments* live.

1. From the ``conda-forge`` channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having a ``conda`` manager installed (see ``conda`` managers in :ref:`conda-install-note`), type in a console:

    .. code-block:: console

        $ conda create -n experiment python={PYTHON_VERSION}
        $ conda activate experiment
        $ conda install -c conda-forge floatcsep

where ``3.9 < {PYTHON_VERSION} <= 3.12`` is at your convenience.

2. From the ``PyPI`` repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having installed the binary dependencies of **pyCSEP**
(see `Installing pyCSEP <https://docs.cseptesting.org/getting_started/installing.html>`_)
and, for pip-only environments, the system dependencies in :ref:`pip-binary-deps`,
install **floatCSEP** by:

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


.. _docker-install:

Docker (Model Containerization & Parallel Run)
----------------------------------------------

Some tutorials and experiments **containerize models with Docker**, and the engine can use Docker
for **parallel** execution. To use these features, please install Docker and (on Linux) perform the
post-installation steps:

- Docker install guide: https://docs.docker.com/engine/install/
- Linux post-installation (non-root usage and add your user to the `docker` group):
  https://docs.docker.com/engine/install/linux-postinstall/

.. tip::
   After installing, verify Docker works:

   .. code-block:: console

      $ docker run --rm hello-world


For Developers
--------------

We recommend using a ``conda`` environment for development to avoid relying on system libraries. For contributing to
the **floatCSEP** codebase, please consider `forking the repository <https://docs.github.com/articles/fork-a-repo>`_
and `creating pull requests <https://docs.github.com/articles/creating-a-pull-request>`_ from there.

    .. code-block:: console

        $ conda create -n floatcsep_dev
        $ conda activate floatcsep_dev
        $ git clone https://github.com/${your_fork}/floatcsep
        $ cd floatcsep
        $ pip install -e ".[dev]"

This will install and configure all the unit-testing, linting, and documentation packages.

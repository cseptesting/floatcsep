Installation
============

    .. note::

        This application uses ``python >= 3.8``

Installing the latest version
-----------------------------

Using ``conda``
~~~~~~~~~~~~~~~

The core of `floatCSEP` is built around the `pyCSEP` package (https://github.com/sceccode/pycsep), which itself contains the core dependencies.

The simplest way to install `floatCSEP`, is by creating a `conda` environment (https://conda.io - checkout Anaconda or Miniconda) and install `pyCSEP` from `conda-forge`

    .. code-block:: console

        $ conda env create -n $NAME
        $ conda activate $NAME
        $ conda install -c conda-forge pycsep

Then, clone and install the floatCSEP source code using ``pip``

    .. code-block:: console

        git clone https://github.com/cseptesting/floatcsep
        cd floatcsep
        pip install .

Using ``apt`` and ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~

To install from ``pip``, we require to install the binary dependencies of ``pyCSEP`` (see `Installing pyCSEP <https://docs.cseptesting.org/getting_started/installing.html>`_}

Then, install the ``pycsep`` latest version

    .. code-block::

        git clone https://github.com/SCECcode/pycsep
        cd pycsep
        python -m virtualenv venv
        source venv/bin/activate
        pip install -e .[all]

and the ``floatcsep`` latest version

    .. code-block::

        cd ..
        git clone https://github.com/cseptesting/floatcsep
        cd floatcsep
        pip install .[all]


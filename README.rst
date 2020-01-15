========
Overview
========

Library for computing solutions to multilevel optimal control problems with differential-algebraic equations

* Free software: BSD 3-Clause License

Installation
============

::

    pip install pymloc

You can also install the in-development version with::

    pip install git+ssh://git@gitlab.tubit.tu-berlin.de/bankmann91/pymloc.git@master

Documentation
=============


https://pymloc.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

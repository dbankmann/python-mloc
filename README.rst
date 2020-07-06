========
Overview
========

Library for computing solutions to multilevel optimal control problems.

Currently linear and strangeness-free differential algebraic equations (DAEs), linear ordinary differential equations (ODEs) and their corresponding linear-quadratic optimal control problems are supported as lower level.

As upper level, nonlinear least squares problems are available.


Purpose
=======

The purpose of this software package is to provide an interface for general multilevel optimization and optimal control problems.
One main goal of this package was to maintain a certain level of abstraction. Implementation of additional features  as new optimization problems or different system classes as, e.g. parameter dependent partial differential equations, parameter dependent partial differential-algebraic equations, or parameter dependent port-Hamiltonian systems should be easily doable without touching the general structure of the code too much.
This, however, should also increase maintainability of the code.

The main building blocks of a multilevel optimization object are variable containers and optimizations. Optimizations depend on lower level variables, higher level variables, and variables of the current level.
Optimizations problems can automatically be turned into *local optimizations* by fixing higher and lower level variables to their current values.
Additionally, optimization objects may possess a sensitivity method, which allows to compute sensitivities with respect to higher level variables.


At the time of handing in this thesis, there are two concrete implementations of optimization problems. First, parameter dependent optimal control problems for strangeness-free differential-algebraic control systems, and, second, nonlinear least squares problems.

The general solution procedure of a multilevel optimal control problem is as follows.
After initialization of all optimization problems and mapping variable containers appropriately, the lowermost optimization is turned into a local optimization by fixing higher level variables. A solution is then iteratively passed to the next optimizations, which are localized and solved again, until the uppermost optimization is reached.
The localized optimizations are *solvable* objects, which are linked to available solvers through a generic interface.





* Free software: BSD 3-Clause License

Installation
============

::

    pip install pymloc

You can also install the in-development version with::

    pip install git+ssh://git@gitlab.tubit.tu-berlin.de/bankmann91/python-mloc.git@master

Documentation
=============


localhost

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

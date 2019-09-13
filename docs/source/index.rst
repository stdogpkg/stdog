.. meta::
    :description lang=en:
        stDoG is a package which can be used to study
        dynamical and structural properties (like spectra) 
        on/off graphs with a large number of vertices.

Home
=====================================

Strucutre and Dyanmics on Graphs
----------------------------------------
.. image:: imgs/stdog.png

The main goal of stDoG is to provide a package which can be used to study
dynamical and structural properties (like spectra) on/off graphs with a large
number of vertices. The modules of stDoG are being built by
combining codes written in *Tensorflow* + *CUDA* and *C++*.


The package is available as as pypi repository

.. code-block:: bash

    $ pip install stdog

The source code is available at <http://github.com/stdogpkg>`.

.. toctree::
   :hidden:

   self

.. toctree::
    :maxdepth: 2
    
    install
    examples

.. toctree::
    :maxdepth: 2
    :caption: Modules:
    
    dynamics
    spectra
    utils    


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

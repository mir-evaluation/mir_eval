.. mir_eval documentation master file

mir_eval Documentation
======================

**mir_eval** is a Python library which provides a transparent, standardized, and straightforward way to evaluate Music Information Retrieval systems.

If you use **mir_eval** in a research project, please cite the following paper:

  C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis, `"mir_eval: A Transparent Implementation of Common MIR Metrics" <http://colinraffel.com/publications/ismir2014mir_eval.pdf>`_, Proceedings of the 15th International Conference on Music Information Retrieval, 2014.
  
Installation
============

The simplest way to install **mir_eval** is via ``pip``:

.. code-block:: console

   python -m pip install mir_eval

If you use `conda` packages, **mir_eval** is available on conda-forge:

.. code-block:: console

   conda install -c conda-forge mir_eval

Alternatively, you can install from source:

.. code-block:: console

   python setup.py install

If you don't use Python and want to get started as quickly as possible, you might consider using `Anaconda <https://www.anaconda.com/download>`_ which makes it easy to install a Python environment which can run **mir_eval**.

Using mir_eval
==============

Once installed, you can import **mir_eval** in your code:

.. code-block:: python

   import mir_eval

For example, to evaluate beat tracking:

.. code-block:: python

   reference_beats = mir_eval.io.load_events('reference_beats.txt')
   estimated_beats = mir_eval.io.load_events('estimated_beats.txt')
   scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)

At the end of execution, ``scores`` will be a dict containing scores 
for all of the metrics implemented in `mir_eval.beat`.  
The keys are metric names and values are the scores achieved.

You can also load in the data, do some preprocessing, and call specific metric functions from the appropriate submodule like so:

.. code-block:: python

   reference_beats = mir_eval.io.load_events('reference_beats.txt')
   estimated_beats = mir_eval.io.load_events('estimated_beats.txt')
   # Crop out beats before 5s, a common preprocessing step
   reference_beats = mir_eval.beat.trim_beats(reference_beats)
   estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
   # Compute the F-measure metric and store it in f_measure
   f_measure = mir_eval.beat.f_measure(reference_beats, estimated_beats)

Alternatively, you can use the evaluator scripts which allow you to run evaluation from the command line, without writing any code.
These scripts are are available here:

https://github.com/craffel/mir_evaluators


API Reference
=============

The structure of the **mir_eval** Python module is as follows:
Each MIR task for which evaluation metrics are included in **mir_eval** is given its own submodule, and each metric is defined as a separate function in each submodule.
Every metric function includes detailed documentation, example usage, input validation, and references to the original paper which defined the metric (see the subsections below).
The task submodules also all contain a function ``evaluate()``, which takes as input reference and estimated annotations and returns a dictionary of scores for all of the metrics implemented (for casual users, this is the place to start).
Finally, each task submodule also includes functions for common data pre-processing steps.

**mir_eval** also includes the following additional submodules:

* :mod:`mir_eval.io` which contains convenience functions for loading in task-specific data from common file formats
* :mod:`mir_eval.util` which includes miscellaneous functionality shared across the submodules
* :mod:`mir_eval.sonify` which implements some simple methods for synthesizing annotations of various formats for "evaluation by ear".
* :mod:`mir_eval.display` which provides functions for plotting annotations for various tasks.

Detailed API documentation for each submodule is available in the API Reference section.
See the :doc:`API Reference <api/index>` for full details.

.. toctree::
   :caption: mir_eval
   :maxdepth: 2

   api/index
   changes

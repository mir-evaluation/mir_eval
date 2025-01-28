**************************
``mir_eval`` Documentation
**************************

``mir_eval`` is a Python library which provides a transparent, standaridized, and straightforward way to evaluate Music Information Retrieval systems.

If you use ``mir_eval`` in a research project, please cite the following paper:

C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis, `"mir_eval: A Transparent Implementation of Common MIR Metrics" <http://colinraffel.com/publications/ismir2014mir_eval.pdf>`_, Proceedings of the 15th International Conference on Music Information Retrieval, 2014.

.. _installation:

Installing ``mir_eval``
=======================

The simplest way to install ``mir_eval`` is by using ``pip``, which will also install the required dependencies if needed.
To install ``mir_eval`` using ``pip``, simply run

``python -m pip install mir_eval``

Alternatively, you can install ``mir_eval`` from source by first installing the dependencies and then running

``python setup.py install``

from the source directory.

If you don't use Python and want to get started as quickly as possible, you might consider using `Anaconda <https://www.anaconda.com/download>`_ which makes it easy to install a Python environment which can run ``mir_eval``.

Using ``mir_eval``
=============================================

Once you've installed ``mir_eval`` (see :ref:`installation`), you can import it in your Python code as follows:

``import mir_eval``

From here, you will typically either load in data and call the ``evaluate()`` function from the appropriate submodule like so::

  reference_beats = mir_eval.io.load_events('reference_beats.txt')
  estimated_beats = mir_eval.io.load_events('estimated_beats.txt')
  # Scores will be a dict containing scores for all of the metrics
  # implemented in mir_eval.beat.  The keys are metric names
  # and values are the scores achieved
  scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)

or you'll load in the data, do some preprocessing, and call specific metric functions from the appropriate submodule like so::

  reference_beats = mir_eval.io.load_events('reference_beats.txt')
  estimated_beats = mir_eval.io.load_events('estimated_beats.txt')
  # Crop out beats before 5s, a common preprocessing step
  reference_beats = mir_eval.beat.trim_beats(reference_beats)
  estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
  # Compute the F-measure metric and store it in f_measure
  f_measure = mir_eval.beat.f_measure(reference_beats, estimated_beats)

The documentation for each metric function, found in the :ref:`mir_eval` section below, contains further usage information.

Alternatively, you can use the evaluator scripts which allow you to run evaluation from the command line, without writing any code.
These scripts are are available here:

https://github.com/craffel/mir_evaluators

.. _mir_eval:

``mir_eval``
============

The structure of the ``mir_eval`` Python module is as follows:
Each MIR task for which evaluation metrics are included in ``mir_eval`` is given its own submodule, and each metric is defined as a separate function in each submodule.
Every metric function includes detailed documentation, example usage, input validation, and references to the original paper which defined the metric (see the subsections below).
The task submodules also all contain a function ``evaluate()``, which takes as input reference and estimated annotations and returns a dictionary of scores for all of the metrics implemented (for casual users, this is the place to start).
Finally, each task submodule also includes functions for common data pre-processing steps.

``mir_eval`` also includes the following additional submodules:

* :mod:`mir_eval.io` which contains convenience functions for loading in task-specific data from common file formats
* :mod:`mir_eval.util` which includes miscellaneous functionality shared across the submodules
* :mod:`mir_eval.sonify` which implements some simple methods for synthesizing annotations of various formats for "evaluation by ear".
* :mod:`mir_eval.display` which provides functions for plotting annotations for various tasks.

The following subsections document each submodule.

:mod:`mir_eval.alignment`
--------------------
.. automodule:: mir_eval.alignment
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.beat`
--------------------
.. automodule:: mir_eval.beat
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.chord`
---------------------
.. automodule:: mir_eval.chord
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.melody`
----------------------
.. automodule:: mir_eval.melody
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.multipitch`
--------------------------
.. automodule:: mir_eval.multipitch
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.onset`
---------------------
.. automodule:: mir_eval.onset
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.pattern`
-----------------------
.. automodule:: mir_eval.pattern
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.segment`
-----------------------
.. automodule:: mir_eval.segment
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.hierarchy`
-------------------------
.. automodule:: mir_eval.hierarchy
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.separation`
--------------------------
.. automodule:: mir_eval.separation
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.tempo`
--------------------------
.. automodule:: mir_eval.tempo
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.transcription`
-----------------------------
.. automodule:: mir_eval.transcription
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.transcription_velocity`
--------------------------------------
.. automodule:: mir_eval.transcription_velocity
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.key`
-----------------------------
.. automodule:: mir_eval.key
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.util`
--------------------
.. automodule:: mir_eval.util
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.io`
------------------
.. automodule:: mir_eval.io
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.sonify`
----------------------
.. automodule:: mir_eval.sonify
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.display`
-----------------------
.. automodule:: mir_eval.display
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Changes
=======
.. toctree::
   :maxdepth: 1

   changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


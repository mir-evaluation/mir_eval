Changes
=======

v0.8.2
------

- `#418`_: Fixed a bug in time-frequency sonification with single-interval inputs.

.. _#418: https://github.com/mir-evaluation/mir_eval/pull/418
   
v0.8.1
------

- `#410`_: Fixed several errors with time-frequency sonification
- `#412`_: Removed unused import of deprecated scipy submodule
- `#416`_: `mir_eval.io` routines now support `pathlib.Path` (and generally `os.Pathlike`) inputs as well as strings or open file descriptors.
- `#416`_: `mir_eval.io.load_wav` is deprecated and will be removed in v0.9.'

.. _#410: https://github.com/mir-evaluation/mir_eval/pull/410
.. _#412: https://github.com/mir-evaluation/mir_eval/pull/412
.. _#416: https://github.com/mir-evaluation/mir_eval/pull/416


v0.8
----

- `#397`_: Removed invalid chord types from `chord.QUALITIES`
- `#390`_: Migrate from personal to organization account
- `#388`_: Update to packaging workflow
- `#387`_: Various modernizations 
- `#385`_: Fixed import error
- `#384`_: Rename `testing` to `tests`
- `#382`_: Deprecated the source separation module
- `#380`_: Matplotlib support modernization
- `#379`_: Support nans in pitch contour sonification
- `#378`_: Improved efficiency in chord validation
- `#374`_: Fixed length calculation error in `sonify.time_frequency`
- `#370`_: Test and CI modernization
- `#367`_: Added interpolation method to docs for `resample_multipitch`
- `#361`_: Added PyPI package release workflow
- `#360`_: Fixed broken link
- `#359`_: Remove support for python 2
- `#356`_, `#357`_, `#358`_: Migrate testing to github actions
- `#355`_: Optimize and fix sonification
- `#353`_: Set numerical precision, fix support for modern numpy

.. _#397: https://github.com/mir-evaluation/mir_eval/pull/397
.. _#390: https://github.com/mir-evaluation/mir_eval/pull/390
.. _#388: https://github.com/mir-evaluation/mir_eval/pull/388
.. _#387: https://github.com/mir-evaluation/mir_eval/pull/387
.. _#385: https://github.com/mir-evaluation/mir_eval/pull/385
.. _#384: https://github.com/mir-evaluation/mir_eval/pull/384
.. _#382: https://github.com/mir-evaluation/mir_eval/pull/382
.. _#380: https://github.com/mir-evaluation/mir_eval/pull/380
.. _#379: https://github.com/mir-evaluation/mir_eval/pull/379
.. _#378: https://github.com/mir-evaluation/mir_eval/pull/378
.. _#374: https://github.com/mir-evaluation/mir_eval/pull/374
.. _#370: https://github.com/mir-evaluation/mir_eval/pull/370
.. _#367: https://github.com/mir-evaluation/mir_eval/pull/367
.. _#361: https://github.com/mir-evaluation/mir_eval/pull/361
.. _#360: https://github.com/mir-evaluation/mir_eval/pull/360
.. _#359: https://github.com/mir-evaluation/mir_eval/pull/359
.. _#356: https://github.com/mir-evaluation/mir_eval/pull/356
.. _#357: https://github.com/mir-evaluation/mir_eval/pull/357
.. _#358: https://github.com/mir-evaluation/mir_eval/pull/358
.. _#355: https://github.com/mir-evaluation/mir_eval/pull/355
.. _#353: https://github.com/mir-evaluation/mir_eval/pull/353



v0.7
----

- `#334`_: Support notation for unknown/ambiguous key or mode
- `#343`_: Add suite of alignment metrics

.. _#334: https://github.com/mir-evaluation/mir_eval/pull/334
.. _#343: https://github.com/mir-evaluation/mir_eval/pull/343

v0.6
----

- `#297`_: Return 0 when no overlap in transcription_velocity
- `#299`_: Allow one reference tempo and both estimate tempi to be zero
- `#301`_: Allow zero tolerance in tempo, but issue a warning
- `#302`_: Loosen separation test tolerance
- `#305`_: Use toarray instead of todense for sparse matrices
- `#307`_: Use tuple index in chord.rotate_bitmap_to_root
- `#309`_: Require matplotlib <3 for testing
- `#312`_: Fix raw chroma accuracy for unvoiced estimates
- `#320`_: Add comment support to io methods
- `#323`_: Fix interpolation in sonify.time_frequency
- `#324`_: Add generalized melody metrics 
- `#327`_: Stop testing 2.7
- `#328`_: Cast n_voiced to int in display.multipitch

.. _#297: https://github.com/mir-evaluation/mir_eval/pull/297
.. _#299: https://github.com/mir-evaluation/mir_eval/pull/299
.. _#301: https://github.com/mir-evaluation/mir_eval/pull/301
.. _#302: https://github.com/mir-evaluation/mir_eval/pull/302
.. _#305: https://github.com/mir-evaluation/mir_eval/pull/305
.. _#307: https://github.com/mir-evaluation/mir_eval/pull/307
.. _#309: https://github.com/mir-evaluation/mir_eval/pull/309
.. _#312: https://github.com/mir-evaluation/mir_eval/pull/312
.. _#320: https://github.com/mir-evaluation/mir_eval/pull/320
.. _#323: https://github.com/mir-evaluation/mir_eval/pull/323
.. _#324: https://github.com/mir-evaluation/mir_eval/pull/324
.. _#327: https://github.com/mir-evaluation/mir_eval/pull/327
.. _#328: https://github.com/mir-evaluation/mir_eval/pull/328

v0.5
----

- `#222`_: added int cast for inferred length in sonify.clicks
- `#225`_: improved t-measures and l-measures 
- `#227`_: added marginal flag to segment.nce
- `#234`_: update display to use matplotlib 2
- `#236`_: force integer division in beat.pscore
- `#240`_: fix unit tests for source separation
- `#242`_: use regexp in chord label validation
- `#245`_: add labeled interval formatter to display
- `#247`_: do not sonify negative amplitudes in time_frequency
- `#249`_: support gaps in util.interpolate_intervals
- `#252`_: add modulo and length arguments to chord.scale_degree_to_bitmap
- `#254`_: fix bss_eval_images single-frame fallback documentation
- `#255`_: fix crackle in sonify.time_frequency
- `#258`_: make util.match_events faster
- `#259`_: run pep8 check after nosetests
- `#263`_: add measures for chord over- and under-segmentation
- `#266`_: add amplitude parameter to sonify.pitch_contour
- `#268`_: update display tests to support mpl2.1
- `#277`_: update requirements and fix deprecations
- `#279`_: isolate matplotlib side effects
- `#282`_: remove evaluator scripts
- `#283`_: add transcription eval with velocity

.. _#222: https://github.com/mir-evaluation/mir_eval/pull/222
.. _#225: https://github.com/mir-evaluation/mir_eval/pull/225
.. _#227: https://github.com/mir-evaluation/mir_eval/pull/227
.. _#234: https://github.com/mir-evaluation/mir_eval/pull/234
.. _#236: https://github.com/mir-evaluation/mir_eval/pull/236
.. _#240: https://github.com/mir-evaluation/mir_eval/pull/240
.. _#242: https://github.com/mir-evaluation/mir_eval/pull/242
.. _#245: https://github.com/mir-evaluation/mir_eval/pull/245
.. _#247: https://github.com/mir-evaluation/mir_eval/pull/247
.. _#249: https://github.com/mir-evaluation/mir_eval/pull/249
.. _#252: https://github.com/mir-evaluation/mir_eval/pull/252
.. _#254: https://github.com/mir-evaluation/mir_eval/pull/254
.. _#255: https://github.com/mir-evaluation/mir_eval/pull/255
.. _#258: https://github.com/mir-evaluation/mir_eval/pull/258
.. _#259: https://github.com/mir-evaluation/mir_eval/pull/259
.. _#263: https://github.com/mir-evaluation/mir_eval/pull/263
.. _#266: https://github.com/mir-evaluation/mir_eval/pull/266
.. _#268: https://github.com/mir-evaluation/mir_eval/pull/268
.. _#277: https://github.com/mir-evaluation/mir_eval/pull/277
.. _#279: https://github.com/mir-evaluation/mir_eval/pull/279
.. _#282: https://github.com/mir-evaluation/mir_eval/pull/282
.. _#283: https://github.com/mir-evaluation/mir_eval/pull/283

v0.4
----

- `#189`_: expanded transcription metrics
- `#195`_: added pitch contour sonification
- `#196`_: added the `display` submodule
- `#203`_: support unsorted segment intervals
- `#205`_: correction in documentation for `sonify.time_frequency`
- `#208`_: refactored file/buffer loading
- `#210`_: added `io.load_tempo`
- `#212`_: added frame-wise blind-source separation evaluation
- `#218`_: speed up `melody.resample_melody_series` when times are equivalent

.. _#189: https://github.com/mir-evaluation/mir_eval/issues/189
.. _#195: https://github.com/mir-evaluation/mir_eval/issues/195
.. _#196: https://github.com/mir-evaluation/mir_eval/issues/196
.. _#203: https://github.com/mir-evaluation/mir_eval/issues/203
.. _#205: https://github.com/mir-evaluation/mir_eval/issues/205
.. _#208: https://github.com/mir-evaluation/mir_eval/issues/208
.. _#210: https://github.com/mir-evaluation/mir_eval/issues/210
.. _#212: https://github.com/mir-evaluation/mir_eval/issues/212
.. _#218: https://github.com/mir-evaluation/mir_eval/pull/218

v0.3
----
- `#170`_: implemented transcription metrics
- `#173`_: fixed a bug in chord sonification
- `#175`_: filter_kwargs passes through `**kwargs`
- `#181`_: added key detection metrics

.. _#170: https://github.com/mir-evaluation/mir_eval/issues/170
.. _#173: https://github.com/mir-evaluation/mir_eval/issues/173
.. _#175: https://github.com/mir-evaluation/mir_eval/issues/175
.. _#181: https://github.com/mir-evaluation/mir_eval/issues/181

v0.2
----

- `#103`_: incomplete files passed to `melody.evaluate` should warn
- `#109`_: `STRICT_BASS_INTERVALS` is now an argument to `chord.encode`
- `#122`_: improved handling of corner cases in beat tracking
- `#136`_: improved test coverage
- `#138`_: PEP8 compliance
- `#139`_: converted documentation to numpydoc style
- `#147`_: fixed a rounding error in segment intervals
- `#150`_: `sonify.chroma` and `sonify.chords` pass `kwargs` to `time_frequecy`
- `#151`_: removed `labels` support from `util.boundaries_to_intervals`
- `#159`_: fixed documentation error in `chord.tetrads`
- `#160`_: fixed documentation error in `util.intervals_to_samples`

.. _#103: https://github.com/mir-evaluation/mir_eval/issues/103
.. _#109: https://github.com/mir-evaluation/mir_eval/issues/109
.. _#122: https://github.com/mir-evaluation/mir_eval/issues/122
.. _#136: https://github.com/mir-evaluation/mir_eval/issues/136
.. _#138: https://github.com/mir-evaluation/mir_eval/issues/138
.. _#139: https://github.com/mir-evaluation/mir_eval/issues/139
.. _#147: https://github.com/mir-evaluation/mir_eval/issues/147
.. _#150: https://github.com/mir-evaluation/mir_eval/issues/150
.. _#151: https://github.com/mir-evaluation/mir_eval/issues/151
.. _#159: https://github.com/mir-evaluation/mir_eval/issues/159
.. _#160: https://github.com/mir-evaluation/mir_eval/issues/160


v0.1
----

- Initial public release.

#!/usr/bin/env python
"""Unit tests for the display module"""

# For testing purposes, clobber the rcfile
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

import pytest

import mir_eval
import mir_eval.display
from mir_eval.io import load_labeled_intervals
from mir_eval.io import load_valued_intervals
from mir_eval.io import load_labeled_events
from mir_eval.io import load_ragged_time_series
from mir_eval.io import load_wav

from packaging import version

# Workaround to enable test skipping on older matplotlibs where we know it to be problematic
MPL_VERSION = version.parse(matplotlib.__version__)
OLD_MPL = not (MPL_VERSION >= version.parse("3.8.0"))

# Workaround for old freetype builds with our image fixtures
FT_VERSION = version.parse(matplotlib.ft2font.__freetype_version__)
OLD_FT = not (FT_VERSION >= version.parse("2.10"))

STYLE = "default"


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_segment"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_segment():
    plt.figure()

    # Load some segment data
    intervals, labels = load_labeled_intervals("data/segment/ref00.lab")

    # Plot the segments with no labels
    mir_eval.display.segments(intervals, labels, text=False)

    # Draw a legend
    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_segment_text"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
@pytest.mark.xfail(OLD_MPL, reason=f"matplotlib version < {MPL_VERSION}", strict=False)
def test_display_segment_text():
    plt.figure()

    # Load some segment data
    intervals, labels = load_labeled_intervals("data/segment/ref00.lab")

    # Plot the segments with no labels
    mir_eval.display.segments(intervals, labels, text=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_labeled_intervals"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_labeled_intervals():
    plt.figure()

    # Load some chord data
    intervals, labels = load_labeled_intervals("data/chord/ref01.lab")

    # Plot the chords with nothing fancy
    mir_eval.display.labeled_intervals(intervals, labels)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_labeled_intervals_noextend"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_labeled_intervals_noextend():
    plt.figure()

    # Load some chord data
    intervals, labels = load_labeled_intervals("data/chord/ref01.lab")

    # Plot the chords with nothing fancy
    ax = plt.axes()
    ax.set_yticklabels([])
    mir_eval.display.labeled_intervals(
        intervals, labels, label_set=[], extend_labels=False, ax=ax
    )
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_labeled_intervals_compare"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_labeled_intervals_compare():
    plt.figure()

    # Load some chord data
    ref_int, ref_labels = load_labeled_intervals("data/chord/ref01.lab")
    est_int, est_labels = load_labeled_intervals("data/chord/est01.lab")

    # Plot reference and estimates using label set extension
    mir_eval.display.labeled_intervals(
        ref_int, ref_labels, alpha=0.5, label="Reference"
    )
    mir_eval.display.labeled_intervals(est_int, est_labels, alpha=0.5, label="Estimate")

    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_labeled_intervals_compare_noextend"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_labeled_intervals_compare_noextend():
    plt.figure()

    # Load some chord data
    ref_int, ref_labels = load_labeled_intervals("data/chord/ref01.lab")
    est_int, est_labels = load_labeled_intervals("data/chord/est01.lab")

    # Plot reference and estimate, but only use the reference labels
    mir_eval.display.labeled_intervals(
        ref_int, ref_labels, alpha=0.5, label="Reference"
    )
    mir_eval.display.labeled_intervals(
        est_int, est_labels, extend_labels=False, alpha=0.5, label="Estimate"
    )

    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_labeled_intervals_compare_common"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_labeled_intervals_compare_common():
    plt.figure()

    # Load some chord data
    ref_int, ref_labels = load_labeled_intervals("data/chord/ref01.lab")
    est_int, est_labels = load_labeled_intervals("data/chord/est01.lab")

    label_set = list(sorted(set(ref_labels) | set(est_labels)))

    # Plot reference and estimate with a common label set
    mir_eval.display.labeled_intervals(
        ref_int, ref_labels, label_set=label_set, alpha=0.5, label="Reference"
    )
    mir_eval.display.labeled_intervals(
        est_int, est_labels, label_set=label_set, alpha=0.5, label="Estimate"
    )

    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_hierarchy_nolabel"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_hierarchy_nolabel():
    plt.figure()

    # Load some chord data
    int0, lab0 = load_labeled_intervals("data/hierarchy/ref00.lab")
    int1, lab1 = load_labeled_intervals("data/hierarchy/ref01.lab")

    # Plot reference and estimate with a common label set
    mir_eval.display.hierarchy([int0, int1], [lab0, lab1])

    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_hierarchy_label"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_hierarchy_label():
    plt.figure()

    # Load some chord data
    int0, lab0 = load_labeled_intervals("data/hierarchy/ref00.lab")
    int1, lab1 = load_labeled_intervals("data/hierarchy/ref01.lab")

    # Plot reference and estimate with a common label set
    mir_eval.display.hierarchy([int0, int1], [lab0, lab1], levels=["Large", "Small"])

    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_pitch_hz"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
def test_display_pitch_hz():
    plt.figure()

    ref_times, ref_freqs = load_labeled_events("data/melody/ref00.txt")
    est_times, est_freqs = load_labeled_events("data/melody/est00.txt")

    # Plot pitches on a Hz scale
    mir_eval.display.pitch(ref_times, ref_freqs, unvoiced=True, label="Reference")
    mir_eval.display.pitch(est_times, est_freqs, unvoiced=True, label="Estimate")
    plt.legend(loc="upper left")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_pitch_midi"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_pitch_midi():
    plt.figure()

    times, freqs = load_labeled_events("data/melody/ref00.txt")

    # Plot pitches on a midi scale with note tickers
    mir_eval.display.pitch(times, freqs, midi=True)
    mir_eval.display.ticker_notes()
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_pitch_midi_hz"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_pitch_midi_hz():
    plt.figure()

    times, freqs = load_labeled_events("data/melody/ref00.txt")

    # Plot pitches on a midi scale with note tickers
    mir_eval.display.pitch(times, freqs, midi=True)
    mir_eval.display.ticker_pitch()
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_multipitch_hz_unvoiced"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_multipitch_hz_unvoiced():
    plt.figure()

    times, pitches = load_ragged_time_series("data/multipitch/est01.txt")

    # Plot pitches on a midi scale with note tickers
    mir_eval.display.multipitch(times, pitches, midi=False, unvoiced=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_multipitch_hz_voiced"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_multipitch_hz_voiced():
    plt.figure()

    times, pitches = load_ragged_time_series("data/multipitch/est01.txt")

    mir_eval.display.multipitch(times, pitches, midi=False, unvoiced=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_multipitch_midi"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_multipitch_midi():
    plt.figure()

    ref_t, ref_p = load_ragged_time_series("data/multipitch/ref01.txt")
    est_t, est_p = load_ragged_time_series("data/multipitch/est01.txt")

    # Plot pitches on a midi scale with note tickers
    mir_eval.display.multipitch(ref_t, ref_p, midi=True, alpha=0.5, label="Reference")
    mir_eval.display.multipitch(est_t, est_p, midi=True, alpha=0.5, label="Estimate")

    plt.legend(loc="upper left")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_piano_roll"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_piano_roll():
    plt.figure()

    ref_t, ref_p = load_valued_intervals("data/transcription/ref04.txt")
    est_t, est_p = load_valued_intervals("data/transcription/est04.txt")

    mir_eval.display.piano_roll(ref_t, ref_p, label="Reference", alpha=0.5)
    mir_eval.display.piano_roll(
        est_t, est_p, label="Estimate", alpha=0.5, facecolor="r"
    )

    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_piano_roll_midi"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_piano_roll_midi():
    plt.figure()

    ref_t, ref_p = load_valued_intervals("data/transcription/ref04.txt")
    est_t, est_p = load_valued_intervals("data/transcription/est04.txt")

    ref_midi = mir_eval.util.hz_to_midi(ref_p)
    est_midi = mir_eval.util.hz_to_midi(est_p)
    mir_eval.display.piano_roll(ref_t, midi=ref_midi, label="Reference", alpha=0.5)
    mir_eval.display.piano_roll(
        est_t, midi=est_midi, label="Estimate", alpha=0.5, facecolor="r"
    )

    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_ticker_midi_zoom"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_ticker_midi_zoom():
    plt.figure()

    plt.plot(np.arange(3))
    mir_eval.display.ticker_notes()
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_separation"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
@pytest.mark.skip()
def test_display_separation():
    plt.figure()

    x0, fs = load_wav("data/separation/ref05/0.wav")
    x1, fs = load_wav("data/separation/ref05/1.wav")
    x2, fs = load_wav("data/separation/ref05/2.wav")

    mir_eval.display.separation([x0, x1, x2], fs=fs)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_separation_label"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
@pytest.mark.xfail(OLD_FT, reason=f"freetype version < {FT_VERSION}", strict=False)
@pytest.mark.skip()
def test_display_separation_label():
    plt.figure()

    x0, fs = load_wav("data/separation/ref05/0.wav")
    x1, fs = load_wav("data/separation/ref05/1.wav")
    x2, fs = load_wav("data/separation/ref05/2.wav")

    mir_eval.display.separation([x0, x1, x2], fs=fs, labels=["Alice", "Bob", "Carol"])

    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_events"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_events():
    plt.figure()

    # Load some event data
    beats_ref = mir_eval.io.load_events("data/beat/ref00.txt")[:30]
    beats_est = mir_eval.io.load_events("data/beat/est00.txt")[:30]

    # Plot both with labels
    mir_eval.display.events(beats_ref, label="reference")
    mir_eval.display.events(beats_est, label="estimate")
    plt.legend(loc="upper right")
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_images=["test_display_labeled_events"],
    extensions=["png"],
    style=STYLE,
    tolerance=6,
)
def test_display_labeled_events():
    plt.figure()

    # Load some event data
    beats_ref = mir_eval.io.load_events("data/beat/ref00.txt")[:10]

    labels = list("abcdefghijklmnop")
    # Plot both with labels
    mir_eval.display.events(beats_ref, labels)
    return plt.gcf()


@pytest.mark.xfail(raises=ValueError)
def test_display_pianoroll_nopitch_nomidi():
    # Issue 214
    mir_eval.display.piano_roll([[0, 1]])

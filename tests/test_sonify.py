"""Unit tests for sonification methods"""

import pytest
import mir_eval
import numpy as np
import scipy


@pytest.mark.parametrize("times", [np.array([1.0]), np.arange(10.0)])
@pytest.mark.parametrize("fs", [8000, 44100])
def test_clicks(times, fs):
    # Test output length for a variety of parameter settings
    click_signal = mir_eval.sonify.clicks(times, fs)
    assert len(click_signal) == times.max() * fs + int(fs * 0.1) + 1
    click_signal = mir_eval.sonify.clicks(times, fs, length=1000)
    assert len(click_signal) == 1000
    click_signal = mir_eval.sonify.clicks(times, fs, click=np.zeros(1000))
    assert len(click_signal) == times.max() * fs + 1000 + 1


@pytest.mark.parametrize("fs", [8000, 44100])
def test_time_frequency(fs):
    # Test length for different inputs
    signal = mir_eval.sonify.time_frequency(
        np.random.standard_normal((100, 1000)),
        np.arange(1, 101),
        np.linspace(0, 10, 1000),
        fs,
    )
    assert len(signal) == 10 * fs

    # Make one longer
    signal = mir_eval.sonify.time_frequency(
        np.random.standard_normal((100, 1000)),
        np.arange(1, 101),
        np.linspace(0, 10, 1000),
        fs,
        length=fs * 11,
    )
    assert len(signal) == 11 * fs

    # Make one shorter
    signal = mir_eval.sonify.time_frequency(
        np.random.standard_normal((100, 1000)),
        np.arange(1, 101),
        np.linspace(0, 10, 1000),
        fs,
        length=fs * 5,
    )
    assert len(signal) == 5 * fs


def test_time_frequency_const():

    # Sonify with a single interval to hit the const interpolator
    s1 = mir_eval.sonify.time_frequency(
        np.ones((1, 1)),
        np.array([60]),
        np.array([[0, 1]]),
        8000,
    )
    # Sonify with two tiem intervals to hit the regular interpolator
    # but the second frequency will have no energy, so it doesn't
    # change the resulting signal
    s2 = mir_eval.sonify.time_frequency(
        np.array([[1, 1], [0, 0]]),
        np.array([60, 90]),
        np.array([[0, 1], [0, 1]]),
        8000,
    )

    assert np.allclose(s1, s2)


def test_time_frequency_offset():
    fs = 8000
    # Length is 3 seconds, first interval starts at 5.
    # Should produce an empty signal.
    signal = mir_eval.sonify.time_frequency(
        np.random.standard_normal((100, 100)),
        np.arange(1, 101),
        np.linspace(5, 10, 100),
        fs,
        length=fs * 3,
    )
    assert len(signal) == 3 * fs
    assert np.allclose(signal, 0)


@pytest.mark.xfail(raises=ValueError)
def test_time_frequency_badtime():
    fs = 8000
    gram = np.ones((10, 20))
    times = np.arange(10)

    mir_eval.sonify.time_frequency(gram, np.arange(1, 11), times, fs)


@pytest.mark.xfail(raises=ValueError)
def test_time_frequency_badintervals():
    fs = 8000
    gram = np.ones((10, 20))
    times = np.ones((11, 2))

    mir_eval.sonify.time_frequency(gram, np.arange(1, 11), times, fs)


@pytest.mark.xfail(raises=ValueError)
def test_time_frequency_frequency():
    fs = 8000
    gram = np.ones((10, 20))
    times = np.arange(20)

    mir_eval.sonify.time_frequency(gram, np.arange(1, 8), times, fs)


@pytest.mark.parametrize("fs", [8000, 44100])
def test_chroma(fs):
    signal = mir_eval.sonify.chroma(
        np.random.standard_normal((12, 1000)), np.linspace(0, 10, 1000), fs
    )
    assert len(signal) == 10 * fs
    signal = mir_eval.sonify.chroma(
        np.random.standard_normal((12, 1000)),
        np.linspace(0, 10, 1000),
        fs,
        length=fs * 11,
    )
    assert len(signal) == 11 * fs


@pytest.mark.parametrize("fs", [8000, 44100])
def test_chords(fs):
    intervals = np.array([np.arange(10), np.arange(1, 11)]).T
    signal = mir_eval.sonify.chords(
        ["C", "C:maj", "D:min7", "E:min", "C#", "C", "C", "C", "C", "C"], intervals, fs
    )
    assert len(signal) == 10 * fs
    signal = mir_eval.sonify.chords(
        ["C", "C:maj", "D:min7", "E:min", "C#", "C", "C", "C", "C", "C"],
        intervals,
        fs,
        length=fs * 11,
    )
    assert len(signal) == 11 * fs


def test_chord_x():
    # This test verifies that X sonifies as silence
    intervals = np.array([[0, 1]])
    signal = mir_eval.sonify.chords(["X"], intervals, 8000)
    assert not np.any(signal), signal


def test_pitch_contour():
    # Generate some random pitch
    fs = 8000
    times = np.linspace(0, 5, num=5 * fs, endpoint=True)

    noise = scipy.ndimage.gaussian_filter1d(np.random.randn(len(times)), sigma=256)
    freqs = 440.0 * 2.0 ** (16 * noise)
    amps = np.linspace(0, 1, num=5 * fs, endpoint=True)

    # negate a bunch of sequences
    idx = np.unique(np.random.randint(0, high=len(times), size=32))
    for start, end in zip(idx[::2], idx[1::2]):
        freqs[start:end] *= -1

    # Test with inferring duration
    x = mir_eval.sonify.pitch_contour(times, freqs, fs)
    assert len(x) == fs * 5

    # Test with an explicit duration
    # This forces the interpolator to go off the end of the sampling grid,
    # which should result in a constant sequence in the output
    x = mir_eval.sonify.pitch_contour(times, freqs, fs, length=fs * 7)
    assert len(x) == fs * 7
    assert np.allclose(x[-fs * 2 :], x[-fs * 2])

    # Test with an explicit duration and a fixed offset
    # This forces the interpolator to go off the beginning of
    # the sampling grid, which should result in a constant output
    x = mir_eval.sonify.pitch_contour(times + 5.0, freqs, fs, length=fs * 7)
    assert len(x) == fs * 7
    assert np.allclose(x[: fs * 5], x[0])

    # Test with explicit amplitude
    x = mir_eval.sonify.pitch_contour(times, freqs, fs, length=fs * 7, amplitudes=amps)
    assert len(x) == fs * 7
    assert np.allclose(x[0], 0)

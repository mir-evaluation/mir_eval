"""
Methods which sonify annotations for "evaluation by ear".
All functions return a raw signal at the specified sampling rate.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import interp1d

from . import util
from . import chord


def clicks(times, fs, click=None, length=None):
    """Return a signal with the signal 'click' placed at each specified time

    Parameters
    ----------
    times : np.ndarray
        times to place clicks, in seconds
    fs : int
        desired sampling rate of the output signal
    click : np.ndarray
        click signal, defaults to a 1 kHz blip
    length : int
        desired number of samples in the output signal,
        defaults to ``times.max()*fs + click.shape[0] + 1``

    Returns
    -------
    click_signal : np.ndarray
        Synthesized click signal

    """
    # Create default click signal
    if click is None:
        # 1 kHz tone, 100ms
        click = np.sin(2 * np.pi * np.arange(fs * 0.1) * 1000 / (1.0 * fs))
        # Exponential decay
        click *= np.exp(-np.arange(fs * 0.1) / (fs * 0.01))
    # Set default length
    if length is None:
        length = int(times.max() * fs + click.shape[0] + 1)

    # Pre-allocate click signal
    click_signal = np.zeros(length)
    # Place clicks
    for time in times:
        # Compute the boundaries of the click
        start = int(time * fs)
        end = start + click.shape[0]
        # Make sure we don't try to output past the end of the signal
        if start >= length:
            break
        if end >= length:
            click_signal[start:] = click[: length - start]
            break
        # Normally, just add a click here
        click_signal[start:end] = click
    return click_signal


def time_frequency(
    gram, frequencies, times, fs, function=np.sin, length=None, n_dec=1, threshold=0.01
):
    r"""Reverse synthesis of a time-frequency representation of a signal

    Parameters
    ----------
    gram : np.ndarray
        ``gram[n, m]`` is the magnitude of ``frequencies[n]``
        from ``times[m]`` to ``times[m + 1]``

        Non-positive magnitudes are interpreted as silence.

    frequencies : np.ndarray
        array of size ``gram.shape[0]`` denoting the frequency (in Hz) of
        each row of gram

    times : np.ndarray, shape= ``(gram.shape[1],)`` or ``(gram.shape[1], 2)``
        Either the start time (in seconds) of each column in the gram,
        or the time interval (in seconds) corresponding to each column.

    fs : int
        desired sampling rate of the output signal

    function : function
        function to use to synthesize notes, should be :math:`2\pi`-periodic

    length : int
        desired number of samples in the output signal,
        defaults to ``times[-1]*fs``

    n_dec : int
        the number of decimals used to approximate each sonfied frequency.
        Defaults to 1 decimal place. Higher precision will be slower.

    threshold : float
        optimizes synthesis to only occur for frequencies that have a
        linear magnitude of at least one element in gram above the given threshold.

    Returns
    -------
    output : np.ndarray
        synthesized version of the piano roll

    """
    # Convert times to intervals if necessary
    time_converted = False
    if times.ndim == 1:
        # Convert to intervals
        times = np.hstack((times[:-1, np.newaxis], times[1:, np.newaxis]))
        # We'll need this to keep track of whether we should pad an interval on
        time_converted = True

    # Default value for length
    if length is None:
        length = int(times[-1, 1] * fs)

    last_time_in_secs = float(length) / fs

    if time_converted and times.shape[0] != gram.shape[1]:
        times = np.vstack((times, [times[-1, 1], last_time_in_secs]))

    if times.shape[0] != gram.shape[1]:
        raise ValueError(
            f"times.shape={times.shape} is incompatible with gram.shape={gram.shape}"
        )

    if frequencies.shape[0] != gram.shape[0]:
        raise ValueError(
            f"frequencies.shape={frequencies.shape} is incompatible with gram.shape={gram.shape}"
        )

    if times.min() > 0:
        # We need to pad a silence column on to gram at the beginning
        gram = np.pad(gram, ((0, 0), (1, 0)), mode="constant")
        times = np.vstack(([0, times.min()], times))

    if times.max() < last_time_in_secs:
        # We need to pad a silence column onto gram at the end
        gram = np.pad(gram, ((0, 0), (0, 1)), mode="constant")
        times = np.vstack((times, [times.max(), last_time_in_secs]))

    # Identify the time intervals that have some overlap with the duration
    idx = np.logical_and(times[:, 1] >= 0, times[:, 0] <= last_time_in_secs)
    gram = gram[:, idx]
    times = np.clip(times[idx], 0, last_time_in_secs)

    # Truncate times so that the shape matches gram. However if the time boundaries were converted
    # to intervals, then the number of times will be reduced by one, so we only truncate
    # if the gram is smaller.
    n_times = min(gram.shape[1], times.shape[0])
    times = times[:n_times]
    # Round up to ensure that the adjusted interval last time does not diverge from length
    # due to a loss of precision and truncation to ints.
    sample_intervals = np.round(times * fs).astype(int)

    # Threshold the tfgram to remove negatives values
    gram = np.maximum(gram, 0)

    # Pre-allocate output signal
    output = np.zeros(length)
    if gram.shape[1] == 0:
        # There are no time intervals to process, so return
        # the empty signal.
        return output

    time_centers = np.mean(times, axis=1) * float(fs)

    # Check if there is at least one element on each frequency that has a value above the threshold
    # to justify processing, for optimisation.
    spectral_max_magnitudes = np.max(gram, axis=1)
    for n, frequency in enumerate(frequencies):
        if spectral_max_magnitudes[n] < threshold:
            continue

        # Get a waveform of length samples at this frequency
        wave = _fast_synthesize(frequency, n_dec, fs, function, length)

        # Interpolate the values in gram over the time grid.
        if len(time_centers) > 1:
            # If times was converted from boundaries to intervals, it will change shape from
            # (len, 1) to (len-1, 2), and hence differ from the length of gram (i.e one less),
            # so we ensure gram is reduced appropriately.
            gram_interpolator = interp1d(
                time_centers,
                gram[n, :n_times],
                kind="linear",
                bounds_error=False,
                fill_value=(gram[n, 0], gram[n, -1]),
            )
        # If only one time point, create constant interpolator
        else:
            gram_interpolator = _const_interpolator(gram[n, 0])

        # Create the time-varying scaling for the entire time interval by the piano roll
        # magnitude and add to the accumulating waveform.
        t_in = max(sample_intervals[0][0], 0)
        t_out = min(sample_intervals[-1][-1], length)
        signal = gram_interpolator(np.arange(t_in, t_out))
        output[t_in:t_out] += wave[: len(signal)] * signal

    # Normalize, but only if there's non-zero values
    norm = np.abs(output).max()
    if norm >= np.finfo(output.dtype).tiny:
        output /= norm

    return output


def _const_interpolator(value):
    """Return a function that returns `value`
    no matter the input.
    """

    def __interpolator(x):
        return value

    return __interpolator


def _fast_synthesize(frequency, n_dec, fs, function, length):
    """Efficiently synthesize a signal.
    Generate one cycle, and simulate arbitrary repetitions
    using array indexing tricks.
    """
    # hack so that we can ensure an integer number of periods and samples
    # rounds frequency to 1st decimal, s.t. 10 * frequency will be an int
    frequency = np.round(frequency, n_dec)

    # Generate 10*frequency periods at this frequency
    # Equivalent to n_samples = int(n_periods * fs / frequency)
    # n_periods = 10*frequency is the smallest integer that guarantees
    # that n_samples will be an integer, since assuming 10*frequency
    # is an integer
    n_samples = int(10.0**n_dec * fs)

    short_signal = function(2.0 * np.pi * np.arange(n_samples) * frequency / fs)

    # Calculate the number of loops we need to fill the duration
    n_repeats = int(np.ceil(length / float(short_signal.shape[0])))

    # Simulate tiling the short buffer by using stride tricks
    long_signal = as_strided(
        short_signal,
        shape=(n_repeats, len(short_signal)),
        strides=(0, short_signal.itemsize),
    )

    # Use a flatiter to simulate a long 1D buffer
    return long_signal.flat


def pitch_contour(
    times, frequencies, fs, amplitudes=None, function=np.sin, length=None, kind="linear"
):
    r"""Sonify a pitch contour.

    Parameters
    ----------
    times : np.ndarray
        time indices for each frequency measurement, in seconds
    frequencies : np.ndarray
        frequency measurements, in Hz.
        Non-positive measurements or NaNs will be interpreted as un-voiced samples.
    fs : int
        desired sampling rate of the output signal
    amplitudes : np.ndarray
        amplitude measurements, nonnegative
        defaults to ``np.ones((length,))``
    function : function
        function to use to synthesize notes, should be :math:`2\pi`-periodic
    length : int
        desired number of samples in the output signal,
        defaults to ``max(times)*fs``
    kind : str
        Interpolation mode for the frequency and amplitude values.
        See: ``scipy.interpolate.interp1d`` for valid settings.

    Returns
    -------
    output : np.ndarray
        synthesized version of the pitch contour
    """
    fs = float(fs)

    if length is None:
        length = int(times.max() * fs)

    # Squash the negative frequencies.
    # wave(0) = 0, so clipping here will un-voice the corresponding instants
    frequencies = np.maximum(frequencies, 0.0)
    # Convert nans to zeros to unvoice
    frequencies = np.nan_to_num(frequencies, copy=False)

    # Build a frequency interpolator
    f_interp = interp1d(
        times * fs,
        2 * np.pi * frequencies / fs,
        kind=kind,
        fill_value=0.0,
        bounds_error=False,
        copy=False,
    )

    # Estimate frequency at sample points
    f_est = f_interp(np.arange(length))

    if amplitudes is None:
        a_est = np.ones((length,))
    else:
        # build an amplitude interpolator
        a_interp = interp1d(
            times * fs,
            amplitudes,
            kind=kind,
            fill_value=0.0,
            bounds_error=False,
            copy=False,
        )
        a_est = a_interp(np.arange(length))

    # Sonify the waveform
    return a_est * function(np.cumsum(f_est))


def chroma(chromagram, times, fs, **kwargs):
    """Reverse synthesis of a chromagram (semitone matrix)

    Parameters
    ----------
    chromagram : np.ndarray, shape=(12, times.shape[0])
        Chromagram matrix, where each row represents a semitone [C->Bb]
        i.e., ``chromagram[3, j]`` is the magnitude of D# from ``times[j]`` to
        ``times[j + 1]``
    times : np.ndarray, shape=(len(chord_labels),) or (len(chord_labels), 2)
        Either the start time of each column in the chromagram,
        or the time interval corresponding to each column.
    fs : int
        Sampling rate to synthesize audio data at
    **kwargs
        Additional keyword arguments to pass to
        :func:`mir_eval.sonify.time_frequency`

    Returns
    -------
    output : np.ndarray
        Synthesized chromagram

    """
    # We'll just use time_frequency with a Shepard tone-gram
    # To create the Shepard tone-gram, we copy the chromagram across 7 octaves
    n_octaves = 7
    # starting from C2
    base_note = 24
    # and weight each octave by a normal distribution
    # The normal distribution has mean 72 (one octave above middle C)
    # and std 6 (one half octave)
    mean = 72
    std = 6
    notes = np.arange(12 * n_octaves) + base_note
    shepard_weight = np.exp(-((notes - mean) ** 2.0) / (2.0 * std**2.0))
    # Copy the chromagram matrix vertically n_octaves times
    gram = np.tile(chromagram.T, n_octaves).T
    # This fixes issues if the supplied chromagram is int type
    gram = gram.astype(float)
    # Apply Sheppard weighting
    gram *= shepard_weight.reshape(-1, 1)
    # Compute frequencies
    frequencies = 440.0 * (2.0 ** ((notes - 69) / 12.0))
    return time_frequency(gram, frequencies, times, fs, **kwargs)


def chords(chord_labels, intervals, fs, **kwargs):
    """Synthesizes chord labels

    Parameters
    ----------
    chord_labels : list of str
        List of chord label strings.
    intervals : np.ndarray, shape=(len(chord_labels), 2)
        Start and end times of each chord label
    fs : int
        Sampling rate to synthesize at
    **kwargs
        Additional keyword arguments to pass to
        :func:`mir_eval.sonify.time_frequency`

    Returns
    -------
    output : np.ndarray
        Synthesized chord labels

    """
    util.validate_intervals(intervals)

    # Convert from labels to chroma
    roots, interval_bitmaps, _ = chord.encode_many(chord_labels)
    chromagram = np.array(
        [
            np.roll(interval_bitmap, root)
            for (interval_bitmap, root) in zip(interval_bitmaps, roots)
        ]
    ).T

    return chroma(chromagram, intervals, fs, **kwargs)

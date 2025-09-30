# CREATED: 4/15/14 9:42 AM by Justin Salamon <justin.salamon@nyu.edu>
"""
Unit tests for mir_eval.melody
"""

import numpy as np
import json
import mir_eval
import glob
import pytest

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/melody/ref*.txt"
EST_GLOB = "data/melody/est*.txt"
SCORES_GLOB = "data/melody/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.fixture
def melody_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    # Load in reference melody
    ref_time, ref_freq = mir_eval.io.load_time_series(ref_f)
    # Load in estimated melody
    est_time, est_freq = mir_eval.io.load_time_series(est_f)
    return ref_time, ref_freq, est_time, est_freq, expected_scores


def test_hz2cents():
    # Unit test some simple values
    hz = np.array([0.0, 10.0, 5.0, 320.0, 1420.31238974231])
    # Expected cent conversion
    expected_cent = np.array([0.0, 0.0, -1200.0, 6000.0, 8580.0773605])
    assert np.allclose(mir_eval.melody.hz2cents(hz), expected_cent)


def test_freq_to_voicing():
    # Unit test some simple values
    hz = np.array([0.0, 100.0, -132.0])
    expected_hz = np.array([0.0, 100.0, 132.0])
    expected_voicing = np.array([0, 1, 0])
    # Check voicing conversion
    res_hz, res_voicing = mir_eval.melody.freq_to_voicing(hz)
    assert np.all(res_hz == expected_hz)
    assert np.all(res_voicing == expected_voicing)

    # Unit test some simple values where voicing is given
    hz = np.array([0.0, 100.0, -132.0, 0, 131.0])
    voicing = np.array([0.8, 0.0, 1.0, 0.0, 0.5])
    expected_hz = np.array([0.0, 100.0, 132.0, 0.0, 131.0])
    expected_voicing = np.array([0.0, 0.0, 1.0, 0.0, 0.5])
    # Check voicing conversion
    res_hz, res_voicing = mir_eval.melody.freq_to_voicing(hz, voicing=voicing)
    assert np.all(res_hz == expected_hz)
    assert np.all(res_voicing == expected_voicing)


def test_constant_hop_timebase():
    hop = 0.1
    end_time = 0.35
    expected_times = np.array([0, 0.1, 0.2, 0.3])
    res_times = mir_eval.melody.constant_hop_timebase(hop, end_time)
    assert np.allclose(res_times, expected_times)


def test_resample_melody_series():
    # Check for a small example including a zero transition
    times = np.arange(4) / 35.0
    cents = np.array([2.0, 0.0, -1.0, 1.0])
    voicing = np.array([1, 0, 1, 1])
    times_new = np.linspace(0, 0.08, 9)
    expected_cents = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0, -0.8, -0.1, 0.6])
    expected_voicing = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
    (res_cents, res_voicing) = mir_eval.melody.resample_melody_series(
        times, cents, voicing, times_new
    )
    assert np.allclose(res_cents, expected_cents)
    assert np.allclose(res_voicing, expected_voicing)

    # Check for a small example including a zero transition - nonbinary voicing
    times = np.arange(4) / 35.0
    cents = np.array([2.0, 0.0, -1.0, 1.0])
    voicing = np.array([0.8, 0.0, 0.2, 1.0])
    times_new = np.linspace(0, 0.08, 9)
    expected_cents = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0, -0.8, -0.1, 0.6])
    expected_voicing = np.array([0.8, 0.52, 0.24, 0.01, 0.08, 0.15, 0.28, 0.56, 0.84])
    (res_cents, res_voicing) = mir_eval.melody.resample_melody_series(
        times, cents, voicing, times_new
    )
    assert np.allclose(res_cents, expected_cents)
    assert np.allclose(res_voicing, expected_voicing)


def test_resample_melody_series_same_times():
    # Check the case where the time bases are identical
    times = np.array([0.0, 0.1, 0.2, 0.3])
    times_new = np.array([0.0, 0.1, 0.2, 0.3])
    cents = np.array([2.0, 0.0, -1.0, 1.0])
    voicing = np.array([0, 0, 1, 1])
    expected_cents = np.array([2.0, 0.0, -1.0, 1.0])
    expected_voicing = np.array([False, False, True, True])
    (res_cents, res_voicing) = mir_eval.melody.resample_melody_series(
        times, cents, voicing, times_new
    )
    assert np.allclose(res_cents, expected_cents)
    assert np.allclose(res_voicing, expected_voicing)

    # Check the case where the time bases are identical - nonbinary voicing
    times = np.array([0.0, 0.1, 0.2, 0.3])
    times_new = np.array([0.0, 0.1, 0.2, 0.3])
    cents = np.array([2.0, 0.0, -1.0, 1.0])
    voicing = np.array([0.5, 0.8, 0.9, 1.0])
    expected_cents = np.array([2.0, 0.0, -1.0, 1.0])
    expected_voicing = np.array([0.5, 0.8, 0.9, 1.0])
    (res_cents, res_voicing) = mir_eval.melody.resample_melody_series(
        times, cents, voicing, times_new
    )
    assert np.allclose(res_cents, expected_cents)
    assert np.allclose(res_voicing, expected_voicing)


def test_to_cent_voicing():
    # We'll just test a few values from one of the test annotations
    ref_file = sorted(glob.glob(REF_GLOB))[0]
    ref_time, ref_freq = mir_eval.io.load_time_series(ref_file)
    est_file = sorted(glob.glob(EST_GLOB))[0]
    est_time, est_freq = mir_eval.io.load_time_series(est_file)
    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
        ref_time, ref_freq, est_time, est_freq
    )
    # Expected values
    test_range = np.arange(220, 225)
    expected_ref_v = np.array([False, False, False, True, True])
    expected_ref_c = np.array([0.0, 0.0, 0.0, 6056.8837818916609, 6028.5504583021921])
    expected_est_v = np.array([False] * 5)
    expected_est_c = np.array([5351.3179423647571] * 5)
    assert np.allclose(ref_v[test_range], expected_ref_v)
    assert np.allclose(ref_c[test_range], expected_ref_c)
    assert np.allclose(est_v[test_range], expected_est_v)
    assert np.allclose(est_c[test_range], expected_est_c)

    # Test that a 0 is added to the beginning
    for return_item in mir_eval.melody.to_cent_voicing(
        np.array([1.0, 2.0]),
        np.array([440.0, 442.0]),
        np.array([1.0, 2.0]),
        np.array([441.0, 443.0]),
    ):
        assert len(return_item) == 3
        assert return_item[0] == return_item[1]

    # Test custom voicings
    ref_time, ref_freq = mir_eval.io.load_time_series(ref_file)
    _, ref_reward = mir_eval.io.load_time_series("data/melody/reward00.txt")
    _, est_voicing = mir_eval.io.load_time_series("data/melody/voicingest00.txt")
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(
        ref_time,
        ref_freq,
        est_time,
        est_freq,
        est_voicing=est_voicing,
        ref_reward=ref_reward,
    )
    # Expected values
    test_range = np.arange(220, 225)
    expected_ref_v = np.array([0.0, 0.0, 0.0, 1.0, 0.3])
    expected_ref_c = np.array([0.0, 0.0, 0.0, 6056.8837818916609, 6028.5504583021921])
    expected_est_v = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    expected_est_c = np.array([5351.3179423647571] * 5)
    assert np.allclose(ref_v[test_range], expected_ref_v)
    assert np.allclose(ref_c[test_range], expected_ref_c)
    assert np.allclose(est_v[test_range], expected_est_v)
    assert np.allclose(est_c[test_range], expected_est_c)


# We can ignore this warning, which occurs when testing with all-zeros reward
@pytest.mark.filterwarnings("ignore:Reference melody has no voiced frames")
def test_continuous_voicing_metrics():
    ref_time = np.array([0.0, 0.1, 0.2, 0.3])
    ref_freq = np.array([440.0, 0.0, 220.0, 220.0])

    est_time = np.array([0.0, 0.1, 0.2, 0.3])
    est_freq = np.array([440.1, 330.0, 440.0, 330.0])

    # test different estimate voicings
    all_est_voicing = [
        np.array([1.0, 0.0, 1.0, 1.0]),  # perfect
        np.array([0.0, 1.0, 0.0, 0.0]),  # all wrong
        np.array([0.5, 0.5, 0.5, 0.5]),  # all 0.5
        np.array([0.8, 0.2, 0.8, 0.8]),  # almost right
        np.array([0.2, 0.8, 0.2, 0.2]),  # almost wrong
    ]

    all_expected = [
        # perfect
        {
            "Voicing Recall": 1.0,
            "Voicing False Alarm": 0.0,
            "Raw Pitch Accuracy": 1.0 / 3.0,
            "Raw Chroma Accuracy": 2.0 / 3.0,
            "Overall Accuracy": 0.5,
        },
        # all wrong
        {
            "Voicing Recall": 0.0,
            "Voicing False Alarm": 1.0,
            "Raw Pitch Accuracy": 1.0 / 3.0,
            "Raw Chroma Accuracy": 2.0 / 3.0,
            "Overall Accuracy": 0.0,
        },
        # all 0.5
        {
            "Voicing Recall": 0.5,
            "Voicing False Alarm": 0.5,
            "Raw Pitch Accuracy": 1.0 / 3.0,
            "Raw Chroma Accuracy": 2.0 / 3.0,
            "Overall Accuracy": 0.25,
        },
        # almost right
        {
            "Voicing Recall": 0.8,
            "Voicing False Alarm": 0.2,
            "Raw Pitch Accuracy": 1.0 / 3.0,
            "Raw Chroma Accuracy": 2.0 / 3.0,
            "Overall Accuracy": 0.4,
        },
        # almost wrong
        {
            "Voicing Recall": 0.2,
            "Voicing False Alarm": 0.8,
            "Raw Pitch Accuracy": 1.0 / 3.0,
            "Raw Chroma Accuracy": 2.0 / 3.0,
            "Overall Accuracy": 0.1,
        },
    ]

    for est_voicing, expected_scores in zip(all_est_voicing, all_expected):
        actual_scores = mir_eval.melody.evaluate(
            ref_time, ref_freq, est_time, est_freq, est_voicing=est_voicing
        )
        for metric in actual_scores:
            assert np.isclose(actual_scores[metric], expected_scores[metric])

    # test different rewards
    all_rewards = [
        np.array([0.5, 0.5, 0.5, 0.5]),  # uniform
        np.array([0.3, 0.3, 0.3, 0.3]),  # uniform - different number
        np.array([0.0, 0.0, 0.0, 0.0]),  # all zero
        np.array([1.0, 0.0, 0.0, 0.0]),  # one weight
        np.array([1.0, 0.0, 1.0, 0.0]),  # two weights
        np.array([1.0, 0.0, 0.5, 0.5]),  # slightly generous
        np.array([0.1, 0.0, 0.1, 0.8]),  # big penalty
    ]
    est_voicing = np.array([1.0, 0.0, 1.0, 1.0])

    all_expected = [
        # uniform
        {
            "Voicing Recall": 1.0,
            "Voicing False Alarm": 0.0,
            "Raw Pitch Accuracy": 1.0 / 3.0,
            "Raw Chroma Accuracy": 2.0 / 3.0,
            "Overall Accuracy": 0.5,
        },
        # uniform - different number
        {
            "Voicing Recall": 1.0,
            "Voicing False Alarm": 0.0,
            "Raw Pitch Accuracy": 1.0 / 3.0,
            "Raw Chroma Accuracy": 2.0 / 3.0,
            "Overall Accuracy": 0.5,
        },
        # all zero
        {
            "Voicing Recall": 1.0,
            "Voicing False Alarm": 0.75,
            "Raw Pitch Accuracy": 0.0,
            "Raw Chroma Accuracy": 0.0,
            "Overall Accuracy": 0.25,
        },
        # one weight
        {
            "Voicing Recall": 1.0,
            "Voicing False Alarm": 2.0 / 3.0,
            "Raw Pitch Accuracy": 1.0,
            "Raw Chroma Accuracy": 1.0,
            "Overall Accuracy": 0.5,
        },
        # two weights
        {
            "Voicing Recall": 1.0,
            "Voicing False Alarm": 0.5,
            "Raw Pitch Accuracy": 0.5,
            "Raw Chroma Accuracy": 1.0,
            "Overall Accuracy": 0.5,
        },
        # slightly generous
        {
            "Voicing Recall": 1.0,
            "Voicing False Alarm": 0.0,
            "Raw Pitch Accuracy": 0.5,
            "Raw Chroma Accuracy": 0.75,
            "Overall Accuracy": 0.625,
        },
        # big penalty
        {
            "Voicing Recall": 1.0,
            "Voicing False Alarm": 0.0,
            "Raw Pitch Accuracy": 0.1,
            "Raw Chroma Accuracy": 0.2,
            "Overall Accuracy": 0.325,
        },
    ]

    for ref_reward, expected_scores in zip(all_rewards, all_expected):
        actual_scores = mir_eval.melody.evaluate(
            ref_time,
            ref_freq,
            est_time,
            est_freq,
            est_voicing=est_voicing,
            ref_reward=ref_reward,
        )
        for metric in actual_scores:
            assert np.isclose(actual_scores[metric], expected_scores[metric])


def test_voicing_measures_empty():
    # We need a special test for voicing_measures because it only takes 2 args
    with pytest.warns() as w:
        # First, test for warnings due to empty voicing arrays
        score = mir_eval.melody.voicing_measures(np.array([]), np.array([]))
    assert len(w) == 4
    assert np.all([issubclass(wrn.category, UserWarning) for wrn in w])
    assert [str(wrn.message) for wrn in w] == [
        "Reference voicing array is empty.",
        "Estimated voicing array is empty.",
        "Reference melody has no voiced frames.",
        "Estimated melody has no voiced frames.",
    ]
    # And that the metric is 0
    assert np.allclose(score, 0)


def test_voicing_measures_unvoiced():
    with pytest.warns() as w:
        # Also test for a warning when the arrays have non-voiced content
        mir_eval.melody.voicing_measures(np.ones(10), np.zeros(10))
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimated melody has no voiced frames."


@pytest.mark.xfail(raises=ValueError)
def test_melody_voicing_badlength():
    # ref and est voicings must be the same length
    mir_eval.melody.voicing_measures(np.ones(10), np.ones(11))


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.melody.raw_pitch_accuracy,
        mir_eval.melody.raw_chroma_accuracy,
        mir_eval.melody.overall_accuracy,
    ],
)
def test_melody_function_empty(metric):
    with pytest.warns() as w:
        # First, test for warnings due to empty voicing arrays
        score = metric(np.array([]), np.array([]), np.array([]), np.array([]))
        assert len(w) == 6
        assert np.all([issubclass(wrn.category, UserWarning) for wrn in w])
        assert [str(wrn.message) for wrn in w] == [
            "Reference voicing array is empty.",
            "Estimated voicing array is empty.",
            "Reference melody has no voiced frames.",
            "Estimated melody has no voiced frames.",
            "Reference frequency array is empty.",
            "Estimated frequency array is empty.",
        ]
        # And that the metric is 0
        assert np.allclose(score, 0)
        # Also test for a warning when the arrays have non-voiced content
        metric(np.ones(10), np.arange(10), np.zeros(10), np.arange(10))
        assert len(w) == 7
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimated melody has no voiced frames."


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.melody.raw_pitch_accuracy,
        mir_eval.melody.raw_chroma_accuracy,
        mir_eval.melody.overall_accuracy,
    ],
)
@pytest.mark.parametrize(
    "ref_freq, est_freq", [(np.ones(11), np.ones(10)), (np.ones(10), np.ones(11))]
)
def test_melody_badlength(metric, ref_freq, est_freq):
    # frequency and time must be the same length
    metric(np.ones(10), ref_freq, np.ones(10), est_freq)


@pytest.mark.parametrize("melody_data", file_sets, indirect=True)
@pytest.mark.parametrize("voicing", [False, True])
def test_melody_functions(melody_data, voicing):
    ref_time, ref_freq, est_time, est_freq, expected_scores = melody_data
    # When voicing=True, do the continuous voicing equivalence check
    if voicing:
        ref_reward = np.ones_like(ref_time)
        est_voicing = (est_freq >= 0).astype(float)
    else:
        ref_reward = None
        est_voicing = None
    scores = mir_eval.melody.evaluate(
        ref_time,
        ref_freq,
        est_time,
        est_freq,
        est_voicing=est_voicing,
        ref_reward=ref_reward,
    )
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)

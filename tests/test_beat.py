"""
Unit tests for mir_eval.beat
"""

import numpy as np
import json
import mir_eval
import glob
import pytest

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/beat/ref*.txt"
EST_GLOB = "data/beat/est*.txt"
SCORES_GLOB = "data/beat/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.fixture
def beat_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    reference_beats = mir_eval.io.load_events(ref_f)
    estimated_beats = mir_eval.io.load_events(est_f)

    return reference_beats, estimated_beats, expected_scores


def test_trim_beats():
    # Construct dummy beat times [0., 1., ...]
    dummy_beats = np.arange(10, dtype=np.float64)
    # We expect trim_beats to remove all beats < 5s
    expected_beats = np.arange(5, 10, dtype=np.float64)
    assert np.allclose(mir_eval.beat.trim_beats(dummy_beats), expected_beats)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.beat.f_measure,
        mir_eval.beat.cemgil,
        mir_eval.beat.goto,
        mir_eval.beat.p_score,
        mir_eval.beat.continuity,
        mir_eval.beat.information_gain,
    ],
)
def test_beat_empty_warnings(metric):
    with pytest.warns(UserWarning, match="Reference beats are empty."):
        metric(np.array([]), np.arange(10))
    with pytest.warns(UserWarning, match="Estimated beats are empty."):
        metric(np.arange(10), np.array([]))
    with pytest.warns(UserWarning, match="beats are empty."):
        assert np.allclose(metric(np.array([]), np.array([])), 0)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.beat.f_measure,
        mir_eval.beat.cemgil,
        mir_eval.beat.goto,
        mir_eval.beat.p_score,
        mir_eval.beat.continuity,
        mir_eval.beat.information_gain,
    ],
)
@pytest.mark.parametrize(
    "beats",
    [
        np.array([[1.0, 2.0]]),  # beats must be a 1d array
        np.array([1e10, 1e11]),  # beats must be not huge
        np.array([2.0, 1.0]),  # must be sorted
    ],
)
@pytest.mark.xfail(raises=ValueError)
def test_beat_fail(metric, beats):
    metric(beats, beats)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.beat.f_measure,
        mir_eval.beat.cemgil,
        mir_eval.beat.goto,
        mir_eval.beat.p_score,
        mir_eval.beat.continuity,
        mir_eval.beat.information_gain,
    ],
)
def test_beat_perfect(metric):
    beats = np.arange(10, dtype=np.float64)
    assert np.allclose(metric(beats, beats), 1)


@pytest.mark.parametrize("beat_data", file_sets, indirect=True)
def test_beat_functions(beat_data):
    reference_beats, estimated_beats, expected_scores = beat_data

    # Compute scores
    scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)
    # Compare them
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)


# Unit tests for specific behavior not covered by the above
def test_goto_proportion_correct():
    # This covers the case when over 75% of the beat tracking is correct, and
    # more than 3 beats are incorrect
    assert mir_eval.beat.goto(
        np.arange(100), np.append(np.arange(80), np.arange(80, 100) + 0.2)
    )


@pytest.mark.parametrize(
    "metric",
    [mir_eval.beat.p_score, mir_eval.beat.continuity, mir_eval.beat.information_gain],
)
def test_warning_on_one_beat(metric):
    # This tests the metrics where passing only a single beat raises a warning
    # and returns 0

    with pytest.warns(UserWarning, match="Only one reference beat"):
        metric(np.array([10]), np.arange(10))
    with pytest.warns(UserWarning, match="Only one estimated beat"):
        metric(np.arange(10), np.array([10]))


def test_continuity_edge_cases():
    # There is some special-case logic for when there are few beats
    assert np.allclose(
        mir_eval.beat.continuity(np.array([6.0, 6.0]), np.array([6.0, 7.0])), 0.0
    )
    assert np.allclose(
        mir_eval.beat.continuity(np.array([6.0, 6.0]), np.array([6.5, 7.0])), 0.0
    )

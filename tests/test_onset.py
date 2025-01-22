"""
Unit tests for mir_eval.onset
"""

import numpy as np
import pytest
import json
import mir_eval
import glob
import warnings

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/onset/ref*.txt"
EST_GLOB = "data/onset/est*.txt"
SCORES_GLOB = "data/onset/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.fixture
def onset_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    reference_onsets = mir_eval.io.load_events(ref_f)
    estimated_onsets = mir_eval.io.load_events(est_f)

    return reference_onsets, estimated_onsets, expected_scores


def test_onset_empty_warnings():
    with pytest.warns(UserWarning, match="Reference onsets are empty."):
        mir_eval.onset.f_measure(np.array([]), np.arange(10))

    with pytest.warns(UserWarning, match="Estimated onsets are empty."):
        mir_eval.onset.f_measure(np.arange(10), np.array([]))

    with pytest.warns(UserWarning, match="onsets are empty"):
        # Also verify that the score is 0
        assert np.allclose(mir_eval.onset.f_measure(np.array([]), np.array([])), 0)


@pytest.mark.xfail(raisses=ValueError)
@pytest.mark.parametrize(
    "onsets",
    [
        np.array([[1.0, 2.0]]),  # must be 1d ndarray
        np.array([1e10, 1e11]),  # must not be huge
        np.array([2.0, 1.0]),  # must be sorted
    ],
)
def test_onset_fail(onsets):
    mir_eval.onset.f_measure(onsets, onsets)


def test_onset_match():
    # Valid onsets which are the same produce a score of 1 for all metrics
    onsets = np.arange(10, dtype=np.float64)
    assert np.allclose(mir_eval.onset.f_measure(onsets, onsets), 1.0)


@pytest.mark.parametrize("onset_data", file_sets, indirect=True)
def test_onset_functions(onset_data):
    reference_onsets, estimated_onsets, expected_scores = onset_data

    # Compute scores
    scores = mir_eval.onset.evaluate(reference_onsets, estimated_onsets)
    # Compare them
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)

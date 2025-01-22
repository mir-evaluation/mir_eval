#!/usr/bin/env python
"""
Unit tests for mir_eval.tempo
"""
import numpy as np
import mir_eval
import json
import glob
import pytest


A_TOL = 1e-12


REF_GLOB = "data/tempo/ref*.lab"
EST_GLOB = "data/tempo/est*.lab"
SCORES_GLOB = "data/tempo/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.fixture
def tempo_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)

    def _load_tempi(filename):
        values = mir_eval.io.load_delimited(filename, [float] * 3)
        return np.concatenate(values[:2]), values[-1][0]

    reference_tempi, ref_weight = _load_tempi(ref_f)
    estimated_tempi, _ = _load_tempi(est_f)

    return reference_tempi, ref_weight, estimated_tempi, expected_scores


def test_zero_tolerance_pass():
    good_ref = np.array([60, 120])
    good_weight = 0.5
    good_est = np.array([120, 180])
    zero_tol = 0.0

    with pytest.warns(
        UserWarning, match="A tolerance of 0.0 may not lead to the results you expect"
    ):
        mir_eval.tempo.detection(good_ref, good_weight, good_est, tol=zero_tol)


def test_tempo_pass():
    good_ref = np.array([60, 120])
    good_weight = 0.5
    good_est = np.array([120, 180])
    good_tol = 0.08

    for good_tempo in [np.array([50, 50]), np.array([0, 50]), np.array([50, 0])]:
        mir_eval.tempo.detection(good_tempo, good_weight, good_est, good_tol)
        mir_eval.tempo.detection(good_ref, good_weight, good_tempo, good_tol)

    # allow both estimates to be zero
    mir_eval.tempo.detection(good_ref, good_weight, np.array([0, 0]), good_tol)


@pytest.mark.xfail(raises=ValueError)
def test_tempo_zero_ref():
    # Both references cannot be zero
    mir_eval.tempo.detection(np.array([0.0, 0.0]), 0.5, np.array([60, 120]))


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("weight", [-1, 1.5])
def test_tempo_weight_range(weight):
    # Weight needs to be in the range [0, 1]
    mir_eval.tempo.detection(np.array([60, 120]), weight, np.array([120, 180]))


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("tol", [-1, 1.5])
def test_tempo_tol_range(tol):
    # Weight needs to be in the range [0, 1]
    mir_eval.tempo.detection(np.array([60, 120]), 0.5, np.array([120, 180]), tol=tol)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "bad_tempo",
    [
        np.array([-1, -1]),
        np.array([-1, 0]),
        np.array([-1, 50]),
        np.array([0, 1, 2]),
        np.array([0]),
    ],
)
def test_tempo_fail_bad_reftempo(bad_tempo):
    good_ref = np.array([60, 120])
    good_est = np.array([120, 180])

    mir_eval.tempo.detection(bad_tempo, 0.5, good_est, 0.08)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "bad_tempo",
    [
        np.array([-1, -1]),
        np.array([-1, 0]),
        np.array([-1, 50]),
        np.array([0, 1, 2]),
        np.array([0]),
    ],
)
def test_tempo_fail_bad_esttempo(bad_tempo):
    good_ref = np.array([60, 120])
    good_est = np.array([120, 180])

    mir_eval.tempo.detection(good_ref, 0.5, bad_tempo, 0.08)


@pytest.mark.parametrize("tempo_data", file_sets, indirect=True)
def test_tempo_regression(tempo_data):
    ref_tempi, ref_weight, est_tempi, expected_scores = tempo_data

    scores = mir_eval.tempo.evaluate(ref_tempi, ref_weight, est_tempi)
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)

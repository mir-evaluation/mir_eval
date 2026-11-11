"""
Tests for mir_eval.key
"""

import glob
import json
import os
import mir_eval
import numpy as np
import pytest

A_TOL = 1e-12

# Path to the fixture files
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REF_GLOB = os.path.join(DATA_DIR, "key", "ref*.txt")
EST_GLOB = os.path.join(DATA_DIR, "key", "est*.txt")
SCORES_GLOB = os.path.join(DATA_DIR, "key", "output*.json")

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0
file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.mark.parametrize(
    "good_key",
    ["C major", "c major", "C# major", "Bb minor", "db minor", "X", "x", "C other"],
)
@pytest.mark.parametrize("bad_key", ["C maj", "Cb major", "C", "K major", "F## minorX other", "x minor"])
def test_key_function_fail(good_key, bad_key):
    score = mir_eval.key.weighted_score(good_key, good_key)
    assert score == 1.0

    with pytest.raises(ValueError):
        mir_eval.key.weighted_score(good_key, bad_key)
    with pytest.raises(ValueError):
        mir_eval.key.weighted_score(bad_key, good_key)


@pytest.fixture
def key_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    reference_key = mir_eval.io.load_key(ref_f)
    estimated_key = mir_eval.io.load_key(est_f)

    return reference_key, estimated_key, expected_scores


@pytest.mark.parametrize("key_data", file_sets, indirect=True)
def test_key_functions(key_data):
    reference_key, estimated_key, expected_scores = key_data
    # Compute scores
    scores = mir_eval.key.evaluate(reference_key, estimated_key)
    # Compare them
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)

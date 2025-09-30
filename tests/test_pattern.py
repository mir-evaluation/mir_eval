"""
Some unit tests for the pattern discovery task.
"""

import numpy as np
import json
import mir_eval
import glob
import pytest

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/pattern/ref*.txt"
EST_GLOB = "data/pattern/est*.txt"
SCORES_GLOB = "data/pattern/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.fixture
def pattern_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    reference_patterns = mir_eval.io.load_patterns(ref_f)
    estimated_patterns = mir_eval.io.load_patterns(est_f)

    return reference_patterns, estimated_patterns, expected_scores


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.pattern.standard_FPR,
        mir_eval.pattern.establishment_FPR,
        mir_eval.pattern.occurrence_FPR,
        mir_eval.pattern.three_layer_FPR,
        mir_eval.pattern.first_n_three_layer_P,
        mir_eval.pattern.first_n_target_proportion_R,
    ],
)
def test_pattern_empty(metric):
    # First, test for a warning on empty pattern
    with pytest.warns(UserWarning, match="Reference patterns are empty"):
        metric([[[]]], [[[(100, 20)]]])

    with pytest.warns(UserWarning, match="Estimated patterns are empty"):
        metric([[[(100, 20)]]], [[[]]])

    with pytest.warns(UserWarning, match="patterns are empty"):
        # And that the metric is 0
        assert np.allclose(metric([[[]]], [[[]]]), 0)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.pattern.standard_FPR,
        mir_eval.pattern.establishment_FPR,
        mir_eval.pattern.occurrence_FPR,
        mir_eval.pattern.three_layer_FPR,
        mir_eval.pattern.first_n_three_layer_P,
        mir_eval.pattern.first_n_target_proportion_R,
    ],
)
@pytest.mark.parametrize(
    "patterns",
    [
        [[[(100, 20)]], []],  # patterns must have at least one occurrence
        [[[(100, 20, 3)]]],  # (onset, midi) tuple must contain 2 elements
    ],
)
@pytest.mark.xfail(raises=ValueError)
def test_pattern_failure(metric, patterns):
    metric(patterns, patterns)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.pattern.standard_FPR,
        mir_eval.pattern.establishment_FPR,
        mir_eval.pattern.occurrence_FPR,
        mir_eval.pattern.three_layer_FPR,
        mir_eval.pattern.first_n_three_layer_P,
        mir_eval.pattern.first_n_target_proportion_R,
    ],
)
def test_pattern_perfect(metric):
    # Valid patterns which are the same produce a score of 1 for all metrics
    patterns = [[[(100, 20), (200, 30)]]]
    assert np.allclose(metric(patterns, patterns), 1)


@pytest.mark.parametrize("pattern_data", file_sets, indirect=True)
def test_pattern_functions(pattern_data):
    reference_patterns, estimated_patterns, expected_scores = pattern_data
    # Compute scores
    scores = mir_eval.pattern.evaluate(reference_patterns, estimated_patterns)
    # Compare them
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)

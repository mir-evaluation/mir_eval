"""
Unit tests for mir_eval.alignment
"""

import glob
import json

import pytest
import numpy as np

import mir_eval

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/alignment/ref*.txt"
EST_GLOB = "data/alignment/est*.txt"
SCORES_GLOB = "data/alignment/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.alignment.absolute_error,
        mir_eval.alignment.percentage_correct,
        mir_eval.alignment.percentage_correct_segments,
        (
            lambda ref_ts, est_ts: mir_eval.alignment.percentage_correct_segments(
                ref_ts, est_ts, duration=max(np.max(ref_ts), np.max(est_ts))
            )
        ),
        mir_eval.alignment.karaoke_perceptual_metric,
    ],
)
@pytest.mark.parametrize(
    "est_alignment, pred_alignment",
    [
        (
            np.array([[1.0, 2.0]]),
            np.array([[1.0, 2.0]]),
        ),  # alignments must be 1d ndarray
        (
            np.array([[-1.0, 2.0]]),
            np.array([[1.0, 2.0]]),
        ),  # alignments must be non-negative
        (np.array([[2.0, 1.0]]), np.array([[1.0, 2.0]])),  # alignments must be sorted
        (
            np.array([[1.0, 2.0]]),
            np.array([[1.0]]),
        ),  # alignments must have the same length
    ],
)
def test_alignment_functions_fail(metric, est_alignment, pred_alignment):
    metric(est_alignment, pred_alignment)


@pytest.fixture
def alignment_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    reference_alignments = mir_eval.io.load_events(ref_f)
    estimated_alignments = mir_eval.io.load_events(est_f)

    return reference_alignments, estimated_alignments, expected_scores


@pytest.mark.parametrize("alignment_data", file_sets, indirect=True)
def test_alignment_functions(alignment_data):
    reference_alignments, estimated_alignments, expected_scores = alignment_data
    scores = mir_eval.alignment.evaluate(reference_alignments, estimated_alignments)

    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)

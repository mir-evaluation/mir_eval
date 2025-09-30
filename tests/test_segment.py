"""
Unit tests for mir_eval.segment
"""

import numpy as np
import json
import mir_eval
import glob
import pytest

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/segment/ref*.lab"
EST_GLOB = "data/segment/est*.lab"
SCORES_GLOB = "data/segment/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.fixture
def segment_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    # Load in an example segmentation annotation
    ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_f)
    # Load in an example segmentation tracker output
    est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_f)

    return ref_intervals, ref_labels, est_intervals, est_labels, expected_scores


@pytest.mark.parametrize(
    "metric", [mir_eval.segment.detection, mir_eval.segment.deviation]
)
def test_segment_boundary_empty(metric):
    with pytest.warns(UserWarning, match="Reference intervals are empty"):
        metric(np.zeros((0, 2)), np.array([[1, 2], [2, 3]]), trim=False)

    with pytest.warns(UserWarning, match="Estimated intervals are empty"):
        metric(np.array([[1, 2], [2, 3]]), np.array([[1, 2]]), trim=True)

    with pytest.warns(UserWarning, match="intervals are empty"):
        empty_intervals = np.zeros((0, 2))
        if metric == mir_eval.segment.detection:
            assert np.allclose(metric(empty_intervals, empty_intervals), 0)
        else:
            assert np.all(np.isnan(metric(empty_intervals, empty_intervals)))


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "metric", [mir_eval.segment.detection, mir_eval.segment.deviation]
)
@pytest.mark.parametrize(
    "intervals",
    [
        # Now test validation function - intervals must be n by 2
        np.array([1, 2, 3, 4]),
        # Interval boundaries must be positive
        np.array([[-1, 2], [2, 3]]),
        # Positive interval durations
        np.array([[2, 1], [2, 3]]),
    ],
)
def test_segment_boundary_errors(metric, intervals):
    metric(intervals, intervals)


def test_segment_boundary_detection_perfect():
    correct_intervals = np.array([[0, 1], [1, 2]])
    assert np.allclose(
        mir_eval.segment.detection(correct_intervals, correct_intervals), 1
    )


def test_segment_boundary_deviation_perfect():
    correct_intervals = np.array([[0, 1], [1, 2]])
    assert np.allclose(
        mir_eval.segment.deviation(correct_intervals, correct_intervals), 0
    )


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.segment.pairwise,
        mir_eval.segment.rand_index,
        mir_eval.segment.ari,
        mir_eval.segment.mutual_information,
        mir_eval.segment.nce,
        mir_eval.segment.vmeasure,
    ],
)
def test_segment_structure_empty(metric):
    with pytest.warns(UserWarning, match="Reference intervals are empty"):
        metric(np.zeros((0, 2)), [], np.array([[0, 1]]), ["foo"])

    with pytest.warns(UserWarning, match="Estimated intervals are empty"):
        metric(np.array([[0, 1]]), ["foo"], np.zeros((0, 2)), [])

    with pytest.warns(UserWarning, match="intervals are empty"):
        empty_intervals = np.zeros((0, 2))
        assert np.allclose(metric(empty_intervals, [], empty_intervals, []), 0)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.segment.pairwise,
        mir_eval.segment.rand_index,
        mir_eval.segment.ari,
        mir_eval.segment.mutual_information,
        mir_eval.segment.nce,
        mir_eval.segment.vmeasure,
    ],
)
@pytest.mark.parametrize(
    "intervals, labels",
    [
        # Test for non-matching numbers of intervals and labels
        (np.array([[2, 1], [2, 3]]), ["a", "b", "c"]),
        # Now test validation function - intervals must be n by 2
        (np.arange(4), ["a", "b", "c", "d"]),
        # Interval boundaries must be positive
        (np.array([[-1, 2], [2, 3]]), ["a", "b"]),
        # Positive interval durations
        (np.array([[2, 1], [2, 3]]), ["a", "b"]),
        # Number of intervals must match number of labels
        (np.array([[2, 1], [2, 3]]), ["a"]),
        # Intervals must start at 0
        (np.array([[1, 2], [2, 3]]), ["a", "b"]),
    ],
)
def test_segment_structure_fail(metric, intervals, labels):
    metric(intervals, labels, intervals, labels)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.segment.pairwise,
        mir_eval.segment.rand_index,
        mir_eval.segment.ari,
        mir_eval.segment.mutual_information,
        mir_eval.segment.nce,
        mir_eval.segment.vmeasure,
    ],
)
def test_segment_structure_end_mismatch(metric):
    reference_intervals = np.array([[0, 1], [1, 2]])
    estimated_intervals = np.array([[0, 1], [1, 3]])
    labels = ["a", "b"]
    metric(reference_intervals, labels, estimated_intervals, labels)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.segment.pairwise,
        mir_eval.segment.rand_index,
        mir_eval.segment.ari,
        mir_eval.segment.mutual_information,
        mir_eval.segment.nce,
        mir_eval.segment.vmeasure,
    ],
)
def test_segment_structure_perfect(metric):
    reference_intervals = np.array([[0, 1], [1, 2]])
    estimated_intervals = np.array([[0, 1], [1, 2]])
    labels = ["a", "b"]
    if metric == mir_eval.segment.mutual_information:
        assert np.allclose(
            metric(reference_intervals, labels, estimated_intervals, labels),
            [np.log(2), 1, 1],
        )
    else:
        assert np.allclose(
            metric(reference_intervals, labels, estimated_intervals, labels), 1
        )


@pytest.mark.parametrize("segment_data", file_sets, indirect=True)
def test_segment_functions(segment_data):
    ref_intervals, ref_labels, est_intervals, est_labels, expected_scores = segment_data

    # Compute scores
    scores = mir_eval.segment.evaluate(
        ref_intervals, ref_labels, est_intervals, est_labels
    )
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)


@pytest.mark.parametrize("segment_data", file_sets, indirect=True)
def test_segment_functions_permuted(segment_data):
    ref_intervals, ref_labels, est_intervals, est_labels, expected_scores = segment_data
    # Also check with permuted references
    idx = np.random.permutation(np.arange(len(ref_intervals)))

    perm_int = ref_intervals[idx]
    perm_lab = [ref_labels[_] for _ in idx]
    scores = mir_eval.segment.evaluate(perm_int, perm_lab, est_intervals, est_labels)
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)

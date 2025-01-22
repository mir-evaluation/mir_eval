"""
Unit tests for mir_eval.hierarchy
"""

import glob
import re

import json

import numpy as np
import scipy.sparse
import mir_eval
import pytest


A_TOL = 1e-12


@pytest.mark.parametrize("window", [5, 10, 15, 30, 90, None])
@pytest.mark.parametrize("frame_size", [0.1, 0.5, 1.0])
def test_tmeasure_pass(window, frame_size):
    # The estimate here gets none of the structure correct.
    ref = [[[0, 30]], [[0, 15], [15, 30]]]
    # convert to arrays
    ref = [np.asarray(_) for _ in ref]

    est = ref[:1]

    # The estimate should get 0 score here
    scores = mir_eval.hierarchy.tmeasure(ref, est, window=window, frame_size=frame_size)

    for k in scores:
        assert k == 0.0

    # The reference should get a perfect score here
    scores = mir_eval.hierarchy.tmeasure(ref, ref, window=window, frame_size=frame_size)

    for k in scores:
        assert k == 1.0


def test_tmeasure_warning():
    # Warn if there are missing boundaries from one layer to the next
    ref = [[[0, 5], [5, 10]], [[0, 10]]]

    ref = [np.asarray(_) for _ in ref]

    with pytest.warns(
        UserWarning, match="Segment hierarchy is inconsistent at level 1"
    ):
        mir_eval.hierarchy.tmeasure(ref, ref)


def test_tmeasure_fail_span():
    # Does not start at 0
    ref = [[[1, 10]], [[1, 5], [5, 10]]]

    ref = [np.asarray(_) for _ in ref]

    with pytest.raises(ValueError):
        mir_eval.hierarchy.tmeasure(ref, ref)

    # Does not end at the right time
    ref = [[[0, 5]], [[0, 5], [5, 6]]]
    ref = [np.asarray(_) for _ in ref]

    with pytest.raises(ValueError):
        mir_eval.hierarchy.tmeasure(ref, ref)

    # Two annotaions of different shape
    ref = [[[0, 10]], [[0, 5], [5, 10]]]
    ref = [np.asarray(_) for _ in ref]

    est = [[[0, 15]], [[0, 5], [5, 15]]]
    est = [np.asarray(_) for _ in est]

    with pytest.raises(ValueError):
        mir_eval.hierarchy.tmeasure(ref, est)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "window, frame_size",
    [(None, -1), (None, 0), (15, -1), (15, 0), (15, 30), (30, -1), (30, 0), (30, 60)],
)
def test_tmeasure_fail_frame_size(window, frame_size):
    ref = [[[0, 60]], [[0, 30], [30, 60]]]
    ref = [np.asarray(_) for _ in ref]

    mir_eval.hierarchy.tmeasure(ref, ref, window=window, frame_size=frame_size)


@pytest.mark.parametrize("frame_size", [0.1, 0.5, 1.0])
def test_lmeasure_pass(frame_size):
    # The estimate here gets none of the structure correct.
    ref = [[[0, 30]], [[0, 15], [15, 30]]]
    ref_lab = [["A"], ["a", "b"]]

    # convert to arrays
    ref = [np.asarray(_) for _ in ref]

    est = ref[:1]
    est_lab = ref_lab[:1]

    # The estimate should get 0 score here
    scores = mir_eval.hierarchy.lmeasure(
        ref, ref_lab, est, est_lab, frame_size=frame_size
    )

    for k in scores:
        assert k == 0.0

    # The reference should get a perfect score here
    scores = mir_eval.hierarchy.lmeasure(
        ref, ref_lab, ref, ref_lab, frame_size=frame_size
    )

    for k in scores:
        assert k == 1.0


def test_lmeasure_warning():
    # Warn if there are missing boundaries from one layer to the next
    ref = [[[0, 5], [5, 10]], [[0, 10]]]

    ref = [np.asarray(_) for _ in ref]
    ref_lab = [["a", "b"], ["A"]]

    with pytest.warns(
        UserWarning, match="Segment hierarchy is inconsistent at level 1"
    ):
        mir_eval.hierarchy.lmeasure(ref, ref_lab, ref, ref_lab)


def test_lmeasure_fail_span():
    # Does not start at 0
    ref = [[[1, 10]], [[1, 5], [5, 10]]]

    ref_lab = [["A"], ["a", "b"]]

    ref = [np.asarray(_) for _ in ref]

    with pytest.raises(ValueError):
        mir_eval.hierarchy.lmeasure(ref, ref_lab, ref, ref_lab)

    # Does not end at the right time
    ref = [[[0, 5]], [[0, 5], [5, 6]]]
    ref = [np.asarray(_) for _ in ref]

    with pytest.raises(ValueError):
        mir_eval.hierarchy.lmeasure(ref, ref_lab, ref, ref_lab)

    # Two annotations of different shape
    ref = [[[0, 10]], [[0, 5], [5, 10]]]
    ref = [np.asarray(_) for _ in ref]

    est = [[[0, 15]], [[0, 5], [5, 15]]]
    est = [np.asarray(_) for _ in est]

    with pytest.raises(ValueError):
        mir_eval.hierarchy.lmeasure(ref, ref_lab, est, ref_lab)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("frame_size", [-1, 0])
def test_lmeasure_fail_frame_size(frame_size):
    ref = [[[0, 60]], [[0, 30], [30, 60]]]
    ref = [np.asarray(_) for _ in ref]
    ref_lab = [["A"], ["a", "b"]]

    mir_eval.hierarchy.lmeasure(ref, ref_lab, ref, ref_lab, frame_size=frame_size)


SCORES_GLOB = "data/hierarchy/output*.json"
sco_files = sorted(glob.glob(SCORES_GLOB))


@pytest.fixture
def hierarchy_outcomes(request):
    sco_f = request.param

    with open(sco_f) as fdesc:
        expected_scores = json.load(fdesc)
    window = float(re.match(r".*output_w=(\d+).json$", sco_f).groups()[0])

    return expected_scores, window


@pytest.mark.parametrize("hierarchy_outcomes", sco_files, indirect=True)
def test_hierarchy_regression(hierarchy_outcomes):
    expected_scores, window = hierarchy_outcomes

    # Hierarchy data is split across multiple lab files for these tests
    ref_files = sorted(glob.glob("data/hierarchy/ref*.lab"))
    est_files = sorted(glob.glob("data/hierarchy/est*.lab"))

    ref_hier = [mir_eval.io.load_labeled_intervals(_) for _ in ref_files]
    est_hier = [mir_eval.io.load_labeled_intervals(_) for _ in est_files]

    ref_ints = [seg[0] for seg in ref_hier]
    ref_labs = [seg[1] for seg in ref_hier]
    est_ints = [seg[0] for seg in est_hier]
    est_labs = [seg[1] for seg in est_hier]

    outputs = mir_eval.hierarchy.evaluate(
        ref_ints, ref_labs, est_ints, est_labs, window=window
    )

    assert outputs.keys() == expected_scores.keys()
    for key in expected_scores:
        assert np.allclose(expected_scores[key], outputs[key], atol=A_TOL)


def test_count_inversions():
    # inversion count = |{(i, j) : a[i] >= b[j]}|
    a = [2, 4, 6]
    b = [1, 2, 3, 4]

    # All inversions (a, b) are:
    # (2, 1), (2, 2)
    # (4, 1), (4, 2), (4, 3), (4, 4)
    # (6, 1), (6, 2), (6, 3), (6, 4)

    assert mir_eval.hierarchy._count_inversions(a, b) == 10

    # All inversions (b, a) are:
    # (2, 2)
    # (3, 2)
    # (4, 2), (4, 4)

    assert mir_eval.hierarchy._count_inversions(b, a) == 4

    # And test with repetitions
    a = [2, 2, 4]
    b = [1, 2, 4, 4]
    # counts: (a, b)
    # (2, 1), (2, 2)
    # (2, 1), (2, 2)
    # (4, 1), (4, 2), (4, 4), (4, 4)

    assert mir_eval.hierarchy._count_inversions(a, b) == 8

    # count: (b, a)
    # (2, 2), (2, 2)
    # (4, 2), (4, 2), (4, 4)
    # (4, 2), (4, 2), (4, 4)

    assert mir_eval.hierarchy._count_inversions(b, a) == 8


def test_meet():
    frame_size = 1
    int_hier = [
        np.array([[0, 10]]),
        np.array([[0, 6], [6, 10]]),
        np.array([[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]),
    ]

    lab_hier = [["X"], ["A", "B"], ["a", "b", "a", "c", "b"]]

    # Target output
    meet_truth = np.asarray(
        [
            [3, 3, 2, 2, 3, 3, 1, 1, 1, 1],  # (XAa)
            [3, 3, 2, 2, 3, 3, 1, 1, 1, 1],  # (XAa)
            [2, 2, 3, 3, 2, 2, 1, 1, 3, 3],  # (XAb)
            [2, 2, 3, 3, 2, 2, 1, 1, 3, 3],  # (XAb)
            [3, 3, 2, 2, 3, 3, 1, 1, 1, 1],  # (XAa)
            [3, 3, 2, 2, 3, 3, 1, 1, 1, 1],  # (XAa)
            [1, 1, 1, 1, 1, 1, 3, 3, 2, 2],  # (XBc)
            [1, 1, 1, 1, 1, 1, 3, 3, 2, 2],  # (XBc)
            [1, 1, 3, 3, 1, 1, 2, 2, 3, 3],  # (XBb)
            [1, 1, 3, 3, 1, 1, 2, 2, 3, 3],  # (XBb)
        ]
    )
    meet = mir_eval.hierarchy._meet(int_hier, lab_hier, frame_size)

    # Is it the right type?
    assert isinstance(meet, scipy.sparse.csr_matrix)
    meet = meet.toarray()

    # Does it have the right shape?
    assert meet.shape == (10, 10)

    # Does it have the right value?
    assert np.all(meet == meet_truth)


def test_compare_frame_rankings():
    # number of pairs (i, j)
    # where ref[i] < ref[j] and est[i] >= est[j]

    ref = np.asarray([1, 2, 3, 3])
    # ref pairs (transitive)
    # (1, 2), (1, 3), (1, 3), (2, 3), (2, 3)
    # ref pairs (non-transitive)
    # (1, 2), (2, 3), (2, 3)

    # Just count the normalizers
    # No self-inversions are possible from ref to itself
    inv, norm = mir_eval.hierarchy._compare_frame_rankings(ref, ref, transitive=True)
    assert inv == 0
    assert norm == 5.0

    inv, norm = mir_eval.hierarchy._compare_frame_rankings(ref, ref, transitive=False)
    assert inv == 0
    assert norm == 3.0

    est = np.asarray([1, 2, 1, 3])
    # In the transitive case, we lose two pairs
    # (1, 3) and (2, 2) -> (1, 1), (2, 1)
    inv, norm = mir_eval.hierarchy._compare_frame_rankings(ref, est, transitive=True)
    assert inv == 2
    assert norm == 5.0

    # In the non-transitive case, we only lose one pair
    # because (1,3) was not counted
    inv, norm = mir_eval.hierarchy._compare_frame_rankings(ref, est, transitive=False)
    assert inv == 1
    assert norm == 3.0

    # Do an all-zeros test
    ref = np.asarray([1, 1, 1, 1])
    inv, norm = mir_eval.hierarchy._compare_frame_rankings(ref, ref, transitive=True)
    assert inv == 0
    assert norm == 0.0

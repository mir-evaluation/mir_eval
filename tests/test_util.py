"""Unit tests for utils"""

import collections

import pytest
import numpy as np
import mir_eval
from mir_eval import util


def test_interpolate_intervals():
    """Check that an interval set is interpolated properly, with boundaries
    conditions and out-of-range values.
    """
    labels = list("abc")
    intervals = np.array([(n, n + 1.0) for n in range(len(labels))])
    time_points = [-1.0, 0.1, 0.9, 1.0, 2.3, 4.0]
    expected_ans = ["N", "a", "a", "b", "c", "N"]
    assert (
        util.interpolate_intervals(intervals, labels, time_points, "N") == expected_ans
    )


def test_interpolate_intervals_gap():
    """Check that an interval set is interpolated properly, with gaps."""
    labels = list("abc")
    intervals = np.array([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])
    time_points = [0.0, 0.75, 1.25, 1.75, 2.25, 2.75, 3.5]
    expected_ans = ["N", "a", "N", "b", "N", "c", "N"]
    assert (
        util.interpolate_intervals(intervals, labels, time_points, "N") == expected_ans
    )


@pytest.mark.xfail(raises=ValueError)
def test_interpolate_intervals_badtime():
    """Check that interpolate_intervals throws an exception if
    input is unordered.
    """
    labels = list("abc")
    intervals = np.array([(n, n + 1.0) for n in range(len(labels))])
    time_points = [-1.0, 0.1, 0.9, 0.8, 2.3, 4.0]
    mir_eval.util.interpolate_intervals(intervals, labels, time_points)


def test_intervals_to_samples():
    """Check that an interval set is sampled properly, with boundaries
    conditions and out-of-range values.
    """
    labels = list("abc")
    intervals = np.array([(n, n + 1.0) for n in range(len(labels))])

    expected_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    expected_labels = ["a", "a", "b", "b", "c", "c"]
    result = util.intervals_to_samples(
        intervals, labels, offset=0, sample_size=0.5, fill_value="N"
    )
    assert result[0] == expected_times
    assert result[1] == expected_labels

    expected_times = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
    expected_labels = ["a", "a", "b", "b", "c", "c"]
    result = util.intervals_to_samples(
        intervals, labels, offset=0.25, sample_size=0.5, fill_value="N"
    )
    assert result[0] == expected_times
    assert result[1] == expected_labels


def test_intersect_files():
    """Check that two non-identical produce correct results."""
    flist1 = ["/a/b/abc.lab", "/c/d/123.lab", "/e/f/xyz.lab"]
    flist2 = ["/g/h/xyz.npy", "/i/j/123.txt", "/k/l/456.lab"]
    sublist1, sublist2 = util.intersect_files(flist1, flist2)
    assert sublist1 == ["/e/f/xyz.lab", "/c/d/123.lab"]
    assert sublist2 == ["/g/h/xyz.npy", "/i/j/123.txt"]
    sublist1, sublist2 = util.intersect_files(flist1[:1], flist2[:1])
    assert sublist1 == []
    assert sublist2 == []


def test_merge_labeled_intervals():
    """Check that two labeled interval sequences merge correctly."""
    x_intvs = np.array([[0.0, 0.44], [0.44, 2.537], [2.537, 4.511], [4.511, 6.409]])
    x_labels = ["A", "B", "C", "D"]
    y_intvs = np.array([[0.0, 0.464], [0.464, 2.415], [2.415, 4.737], [4.737, 6.409]])
    y_labels = [0, 1, 2, 3]
    expected_intvs = [
        [0.0, 0.44],
        [0.44, 0.464],
        [0.464, 2.415],
        [2.415, 2.537],
        [2.537, 4.511],
        [4.511, 4.737],
        [4.737, 6.409],
    ]
    expected_x_labels = ["A", "B", "B", "B", "C", "D", "D"]
    expected_y_labels = [0, 0, 1, 2, 2, 2, 3]
    new_intvs, new_x_labels, new_y_labels = util.merge_labeled_intervals(
        x_intvs, x_labels, y_intvs, y_labels
    )

    assert new_x_labels == expected_x_labels
    assert new_y_labels == expected_y_labels
    assert new_intvs.tolist() == expected_intvs

    # Check that invalid inputs raise a ValueError
    y_intvs[-1, -1] = 10.0
    with pytest.raises(ValueError):
        util.merge_labeled_intervals(x_intvs, x_labels, y_intvs, y_labels)


def test_boundaries_to_intervals():
    # Basic tests
    boundaries = np.arange(10)
    correct_intervals = np.array([np.arange(10 - 1), np.arange(1, 10)]).T
    intervals = mir_eval.util.boundaries_to_intervals(boundaries)
    assert np.all(intervals == correct_intervals)


def test_adjust_events():
    # Test appending at the end
    events = np.arange(1, 11)
    labels = [str(n) for n in range(10)]
    new_e, new_l = mir_eval.util.adjust_events(events, labels, 0.0, 11.0)
    assert new_e[0] == 0.0
    assert new_l[0] == "__T_MIN"
    assert new_e[-1] == 11.0
    assert new_l[-1] == "__T_MAX"
    assert np.all(new_e[1:-1] == events)
    assert new_l[1:-1] == labels

    # Test trimming
    new_e, new_l = mir_eval.util.adjust_events(events, labels, 0.0, 9.0)
    assert new_e[0] == 0.0
    assert new_l[0] == "__T_MIN"
    assert new_e[-1] == 9.0
    assert np.all(new_e[1:] == events[:-1])
    assert new_l[1:] == labels[:-1]


def test_bipartite_match():
    # This test constructs a graph as follows:
    #   v9 -- (u0)
    #   v8 -- (u0, u1)
    #   v7 -- (u0, u1, u2)
    #   ...
    #   v0 -- (u0, u1, ..., u9)
    #
    # This structure and ordering of this graph should force Hopcroft-Karp to
    # hit each algorithm/layering phase
    #
    G = collections.defaultdict(list)

    u_set = [f"u{_:d}" for _ in range(10)]
    v_set = [f"v{_:d}" for _ in range(len(u_set) + 1)]
    for i, u in enumerate(u_set):
        for v in v_set[: -i - 1]:
            G[v].append(u)

    matching = util._bipartite_match(G)

    # Make sure that each u vertex is matched
    assert len(matching) == len(u_set)

    # Make sure that there are no duplicate keys
    lhs = {k for k in matching}
    rhs = {matching[k] for k in matching}

    assert len(matching) == len(lhs)
    assert len(matching) == len(rhs)

    # Finally, make sure that all detected edges are present in G
    for k in matching:
        v = matching[k]
        assert v in G[k] or k in G[v]


def test_outer_distance_mod_n():
    ref = [1.0, 2.0, 3.0]
    est = [1.1, 6.0, 1.9, 5.0, 10.0]
    expected = np.array(
        [
            [0.1, 5.0, 0.9, 4.0, 3.0],
            [0.9, 4.0, 0.1, 3.0, 4.0],
            [1.9, 3.0, 1.1, 2.0, 5.0],
        ]
    )
    actual = mir_eval.util._outer_distance_mod_n(ref, est)
    assert np.allclose(actual, expected)

    ref = [13.0, 14.0, 15.0]
    est = [1.1, 6.0, 1.9, 5.0, 10.0]
    expected = np.array(
        [
            [0.1, 5.0, 0.9, 4.0, 3.0],
            [0.9, 4.0, 0.1, 3.0, 4.0],
            [1.9, 3.0, 1.1, 2.0, 5.0],
        ]
    )
    actual = mir_eval.util._outer_distance_mod_n(ref, est)
    assert np.allclose(actual, expected)


def test_match_events():
    ref = [1.0, 2.0, 3.0]
    est = [1.1, 6.0, 1.9, 5.0, 10.0]
    expected = [(0, 0), (1, 2)]
    actual = mir_eval.util.match_events(ref, est, 0.5)
    assert actual == expected

    ref = [1.0, 2.0, 3.0, 11.9]
    est = [1.1, 6.0, 1.9, 5.0, 10.0, 0.0]
    expected = [(0, 0), (1, 2), (3, 5)]
    actual = mir_eval.util.match_events(
        ref, est, 0.5, distance=mir_eval.util._outer_distance_mod_n
    )
    assert actual == expected


def test_fast_hit_windows():
    ref = [1.0, 2.0, 3.0]
    est = [1.1, 6.0, 1.9, 5.0, 10.0]

    ref_fast, est_fast = mir_eval.util._fast_hit_windows(ref, est, 0.5)
    ref_slow, est_slow = np.where(np.abs(np.subtract.outer(ref, est)) <= 0.5)

    assert np.all(ref_fast == ref_slow)
    assert np.all(est_fast == est_slow)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "intervals",
    [
        # Test for ValueError when interval shape is invalid
        np.array([[1.0], [2.5], [5.0]]),
        # Test for ValueError when times are negative
        np.array([[1.0, -2.0], [2.5, 3.0], [5.0, 6.0]]),
        # Test for ValueError when duration is zero
        np.array([[1.0, 2.0], [2.5, 2.5], [5.0, 6.0]]),
        # Test for ValueError when duration is negative
        np.array([[1.0, 2.0], [2.5, 1.5], [5.0, 6.0]]),
    ],
)
def test_validate_intervals(intervals):
    mir_eval.util.validate_intervals(intervals)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "events",
    [
        # Test for ValueError when max_time is violated
        np.array([100.0, 100000.0]),
        # Test for ValueError when events aren't 1-d arrays
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        # Test for ValueError when event times are not increasing
        np.array([1.0, 2.0, 5.0, 3.0]),
    ],
)
def test_validate_events(events):
    mir_eval.util.validate_events(events)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "freqs",
    [
        # Test for ValueError when max_freq is violated
        np.array([100, 10000]),
        # Test for ValueError when min_freq is violated
        np.array([2, 200]),
        # Test for ValueError when events aren't 1-d arrays
        np.array([[100, 200], [300, 400]]),
        # Test for ValueError when allow_negatives is false and negative values
        # are passed
        np.array([-100, 200]),
    ],
)
def test_validate_frequencies(freqs):
    mir_eval.util.validate_frequencies(freqs, 5000, 20, allow_negatives=False)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "freqs",
    [
        # Test for ValueError when max_freq is violated and allow_negatives=True
        np.array([100, -100000]),
        # Test for ValueError when min_freq is violated and allow_negatives=True
        np.array([-2, 200]),
    ],
)
def test_validate_frequencies_negative(freqs):
    mir_eval.util.validate_frequencies(freqs, 5000, 20, allow_negatives=True)


def test_has_kwargs():
    def __test(target, f):
        assert target == mir_eval.util.has_kwargs(f)

    def f1(_):
        return None

    def f2(_=5):
        return None

    def f3(*_):
        return None

    def f4(_, **kw):
        return None

    def f5(_=5, **kw):
        return None

    assert not mir_eval.util.has_kwargs(f1)
    assert not mir_eval.util.has_kwargs(f2)
    assert not mir_eval.util.has_kwargs(f3)
    assert mir_eval.util.has_kwargs(f4)
    assert mir_eval.util.has_kwargs(f5)


@pytest.mark.parametrize(
    "x,labels,x_true,lab_true",
    [
        (
            np.asarray([[10, 20], [0, 10]]),
            ["a", "b"],
            np.asarray([[0, 10], [10, 20]]),
            ["b", "a"],
        ),
        (
            np.asarray([[0, 10], [10, 20]]),
            ["b", "a"],
            np.asarray([[0, 10], [10, 20]]),
            ["b", "a"],
        ),
    ],
)
def test_sort_labeled_intervals_with_labels(x, labels, x_true, lab_true):
    xs, ls = mir_eval.util.sort_labeled_intervals(x, labels)
    assert np.allclose(xs, x_true)
    assert ls == lab_true


@pytest.mark.parametrize(
    "x,x_true",
    [
        (np.asarray([[10, 20], [0, 10]]), np.asarray([[0, 10], [10, 20]])),
        (np.asarray([[0, 10], [10, 20]]), np.asarray([[0, 10], [10, 20]])),
    ],
)
def test_sort_labeled_intervals_without_labels(x, x_true):
    xs = mir_eval.util.sort_labeled_intervals(x)
    assert np.allclose(xs, x_true)

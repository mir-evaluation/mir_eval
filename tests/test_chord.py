"""
Unit tests for mir_eval.chord
"""

import mir_eval
import numpy as np
import pytest
import glob
import json

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/chord/ref*.lab"
EST_GLOB = "data/chord/est*.lab"
SCORES_GLOB = "data/chord/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.fixture
def chord_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    # Load in reference melody
    ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_f)
    # Load in estimated melody
    est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_f)
    return ref_intervals, ref_labels, est_intervals, est_labels, expected_scores


@pytest.mark.parametrize(
    "pitch, semitone", [("Gbb", 5), ("G", 7), ("G#", 8), ("Cb", 11), ("B#", 0)]
)
def test_pitch_class_to_semitone_valid(pitch, semitone):
    assert mir_eval.chord.pitch_class_to_semitone(pitch) == semitone


@pytest.mark.parametrize("pitch", ["Cab", "#C", "bG"])
@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_pitch_class_to_semitone_fail(pitch):
    mir_eval.chord.pitch_class_to_semitone(pitch)


@pytest.mark.parametrize(
    "degree, semitone",
    [
        ("b7", 10),
        ("#3", 5),
        ("1", 0),
        ("b1", -1),
        ("#7", 12),
        ("bb5", 5),
        ("11", 17),
        ("#13", 22),
    ],
)
def test_scale_degree_to_semitone(degree, semitone):
    assert mir_eval.chord.scale_degree_to_semitone(degree) == semitone


@pytest.mark.parametrize("degree", ["7b", "4#", "77", "15"])
@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_scale_degree_to_semitone(degree):
    mir_eval.chord.scale_degree_to_semitone(degree)


@pytest.mark.parametrize(
    "degree, bitmap, modulo, length",
    [
        ("3", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], True, 12),
        ("*3", [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0], True, 12),
        ("b1", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], True, 12),
        ("9", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], True, 12),
        ("9", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], False, 12),
        ("9", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], False, 15),
    ],
)
def test_scale_degree_to_bitmap(degree, bitmap, modulo, length):
    assert np.allclose(
        mir_eval.chord.scale_degree_to_bitmap(degree, modulo=modulo, length=length),
        bitmap,
    )


@pytest.mark.parametrize(
    "label", ["C", "Eb:min/5", "A#:dim7", "B:maj(*1,*5)/3", "A#:sus4", "A:(9,11)"]
)
def test_validate_chord_label(label):
    # For valid labels, calling the function without an error = pass
    mir_eval.chord.validate_chord_label(label)


@pytest.mark.parametrize(
    "label", ["C::maj", "C//5", "C((4)", "C5))", "C:maj(*3/3", "Cmaj*3/3)", "asdf"]
)
@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_validate_bad_chord_label(label):
    mir_eval.chord.validate_chord_label(label)


@pytest.mark.parametrize(
    "label, split",
    [
        ("C", ["C", "maj", set(), "1"]),
        ("B:maj(*1,*3)/5", ["B", "maj", {"*1", "*3"}, "5"]),
        ("Ab:min/b3", ["Ab", "min", set(), "b3"]),
        ("N", ["N", "", set(), ""]),
        ("G:(3)", ["G", "", {"3"}, "1"]),
    ],
)
def test_split(label, split):
    assert mir_eval.chord.split(label) == split


@pytest.mark.parametrize(
    "label, split",
    [("C", ["C", "maj", set(), "1"]), ("C:minmaj7", ["C", "min", {"7"}, "1"])],
)
def test_split_extended(label, split):
    # Test with reducing extended chords
    mir_eval.chord.split(label, reduce_extended_chords=True) == split


@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_split_fail():
    # Test that an exception is raised when a chord with an omission but no
    # quality is supplied
    mir_eval.chord.split("C(*5)")


# Arguments are root, quality, extensions, bass
@pytest.mark.parametrize(
    "label, split",
    [
        ("F#", ("F#", "", None, "")),
        ("F#:hdim7", ("F#", "hdim7", None, "")),
        ("F#:(*b3,4)", ("F#", "", ["*b3", "4"], "")),
        ("F#/b7", ("F#", "", None, "b7")),
        ("F#:(*b3,4)/b7", ("F#", "", ["*b3", "4"], "b7")),
        ("F#:hdim7/b7", ("F#", "hdim7", None, "b7")),
        ("F#:hdim7(*b3,4)/b7", ("F#", "hdim7", ["*b3", "4"], "b7")),
    ],
)
def test_join(label, split):
    # Test is relying on implicit parameter ordering here: root, quality, extensions, bass
    assert mir_eval.chord.join(*split) == label


@pytest.mark.parametrize(
    "bitmap, root, expected_bitmap",
    [
        ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 0, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
        ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 5, [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
        (
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            11,
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        ),
    ],
)
def test_rotate_bitmaps_to_roots(bitmap, root, expected_bitmap):
    ans = mir_eval.chord.rotate_bitmaps_to_roots([bitmap], [root])
    assert np.all(ans == [expected_bitmap])


@pytest.mark.parametrize(
    "label, e_root, e_interval, e_bass, reduce, strict",
    [
        ("B:maj(*1,*3)/5", 11, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 7, False, False),
        ("G:dim", 7, [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], 0, False, False),
        ("C:(3)/3", 0, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 4, False, False),
        ("A:9/b3", 9, [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], 3, False, False),
        ("G:dim(4)/6", 7, [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], 9, False, False),
        ("A:9", 9, [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], 0, True, False),
    ],
)
def test_chord_encode(label, e_root, e_interval, e_bass, reduce, strict):
    root, intervals, bass = mir_eval.chord.encode(
        label, reduce_extended_chords=reduce, strict_bass_intervals=strict
    )
    assert root == e_root, (root, e_root)
    assert np.all(intervals == e_interval), (intervals, e_interval)
    assert bass == e_bass, (bass, e_bass)


@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_chord_encode_fail():
    # Non-chord bass notes *must* be explicitly named as extensions when
    #   strict_bass_intervals == True
    mir_eval.chord.encode(
        "G:dim(4)/6", reduce_extended_chords=False, strict_bass_intervals=True
    )


def test_encode_many():
    """Does all of the logic for checking encode_many"""
    labels = ["B:maj(*1,*3)/5", "B:maj(*1,*3)/5", "N", "C:min", "C:min"]
    expected_roots = [11, 11, -1, 0, 0]
    expected_intervals = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    ]
    expected_basses = [7, 7, -1, 0, 0]

    roots, intervals, basses = mir_eval.chord.encode_many(labels)
    assert np.all(roots == expected_roots)
    assert np.all(intervals == expected_intervals)
    assert np.all(basses == expected_basses)


def __check_one_metric(metric, ref_label, est_label, score):
    """Checks that a metric function produces score given ref_label and
    est_label"""
    # We provide a dummy interval.  We're just checking one pair
    # of labels at a time.
    assert metric([ref_label], [est_label]) == score


def __check_not_comparable(metric, ref_label, est_label):
    """Checks that ref_label is not comparable to est_label by metric"""
    # Try to produce the warning
    with pytest.warns(
        UserWarning,
        match="No reference chords were comparable to estimated chords, returning 0.",
    ):
        score = mir_eval.chord.weighted_accuracy(
            metric([ref_label], [est_label]), np.array([1])
        )
        # And confirm that the metric is 0
        assert np.allclose(score, 0)


# TODO(ejhumphrey): Comparison functions lacking unit tests.
# test_root()


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("N", "N", 1.0),
        ("C:maj", "N", 0.0),
        ("C:maj", "C:aug", 0.0),
        ("C:maj", "C:dim", 0.0),
        ("C:min", "C:dim", 0.0),
        ("C:maj", "C:5", 0.0),
        ("C:maj", "C:sus4", 0.0),
        ("G:min", "G:sus2", 0.0),
        ("C:maj", "G:maj", 0.0),
        ("C:min", "C:hdim7", 0.0),
        ("C:min", "C:min7", 1.0),
        ("C:maj", "C:maj6", 1.0),
        ("F:maj", "F:min6", 0.0),
        ("C:maj7", "C:minmaj7", 1.0),
        ("A:maj", "A:7", 1.0),
        ("A:maj", "A:9", 1.0),
    ],
)
def test_mirex(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.mirex, ref_label, est_label, score)


@pytest.mark.parametrize("ref, est", [("C:5", "C:maj"), ("X", "N")])
def test_mirex_nocomp(ref, est):
    __check_not_comparable(mir_eval.chord.mirex, ref, est)


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("N", "N", 1.0),
        ("C:maj", "N", 0.0),
        ("C:maj", "C:aug", 1.0),
        ("C:maj", "C:dim", 0.0),
        ("C:min", "C:dim", 1.0),
        ("C:maj", "C:sus4", 1.0),
        ("G:min", "G:sus2", 0.0),
        ("C:maj", "G:maj", 0.0),
        ("C:min", "C:hdim7", 1.0),
        ("C:min", "C:min7", 1.0),
        ("C:maj", "C:maj6", 1.0),
        ("F:maj", "F:min6", 0.0),
        ("C:maj", "C:minmaj7", 0.0),
        ("A:maj", "A:7", 1.0),
        ("A:maj", "A:9", 1.0),
    ],
)
def test_thirds(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.thirds, ref_label, est_label, score)


def test_thirds_nocomp():
    __check_not_comparable(mir_eval.chord.thirds, "X", "N")


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("C:maj/5", "C:sus4/5", 1.0),
        ("G:min", "G:min/b3", 0.0),
        ("C:maj", "C:maj/5", 0.0),
        ("C:min/b3", "C:hdim7/b3", 1.0),
        ("C:min", "C:dim", 1.0),
    ],
)
def test_thirds_inv(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.thirds_inv, ref_label, est_label, score)


def test_thirds_inv_nocomp():
    __check_not_comparable(mir_eval.chord.thirds_inv, "X", "N")


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("C:min", "C:min7", 1.0),
        ("C:maj", "C:7", 1.0),
        ("C:maj", "C:aug", 0.0),
        ("C:min", "C:dim", 0.0),
        ("C:maj", "C:sus2", 0.0),
        ("C:maj", "C:sus4", 0.0),
        ("G:min", "G:minmaj7", 1.0),
        ("C:maj", "G:maj", 0.0),
        ("C:min", "C:hdim7", 0.0),
        ("C:min", "C:min6", 1.0),
    ],
)
def test_triads(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.triads, ref_label, est_label, score)


def test_triads_nocomp():
    __check_not_comparable(mir_eval.chord.triads, "X", "N")


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("C:maj/5", "C:maj7/5", 1.0),
        ("G:min", "G:min7/5", 0.0),
        ("C:maj", "C:7/5", 0.0),
        ("C:min/b3", "C:min6/b3", 1.0),
        ("C:min/b3", "C:dim/b3", 0.0),
    ],
)
def test_triads_inv(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.triads_inv, ref_label, est_label, score)


def test_triads_inv_nocomp():
    __check_not_comparable(mir_eval.chord.triads_inv, "X", "N")


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("C:min", "C:min7", 0.0),
        ("C:maj", "C:maj6", 0.0),
        ("C:7", "C:9", 1.0),
        ("C:maj7", "C:maj7/5", 1.0),
        ("C:sus2", "C:sus2/2", 1.0),
        ("C:7/3", "C:11/b7", 1.0),
        ("G:min", "G:sus2", 0.0),
        ("C:maj", "G:maj", 0.0),
        ("C:min", "C:hdim7", 0.0),
        ("C:min", "C:minmaj7", 0.0),
    ],
)
def test_tetrads(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.tetrads, ref_label, est_label, score)


def test_tetrads_nocomp():
    __check_not_comparable(mir_eval.chord.tetrads, "X", "N")


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("C:maj7/5", "C:maj7/3", 0.0),
        ("G:min", "G:min/b3", 0.0),
        ("C:7/5", "C:13/5", 1.0),
        ("C:min/b3", "C:hdim7/b3", 0.0),
        ("C:min9", "C:min7", 1.0),
    ],
)
def test_tetrads_inv(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.tetrads_inv, ref_label, est_label, score)


def test_tetrads_inv_nocomp():
    __check_not_comparable(mir_eval.chord.tetrads_inv, "X", "N")


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("N", "N", 1.0),
        ("C:maj", "N", 0.0),
        ("C:maj", "C:aug", 0.0),
        ("C:min", "C:dim", 0.0),
        ("G:maj7", "G", 1.0),
    ],
)
def test_majmin(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.majmin, ref_label, est_label, score)


@pytest.mark.parametrize("ref_label, est_label", [("C:aug", "C:maj"), ("X", "N")])
def test_majmin_nocomp(ref_label, est_label):
    __check_not_comparable(mir_eval.chord.majmin, ref_label, est_label)


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("C:maj/5", "C:sus4/5", 0.0),
        ("G:min", "G:min/b3", 0.0),
        ("C:maj/5", "C:maj/5", 1.0),
        ("C:min7", "C:min", 1.0),
        ("G:min/b3", "G:min/b3", 1.0),
        ("C:maj7/5", "C:maj/5", 1.0),
        ("C:7", "C:maj", 1.0),
    ],
)
def test_majmin_inv(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.majmin_inv, ref_label, est_label, score)


@pytest.mark.parametrize(
    "ref_label, est_label",
    [
        ("C:hdim7/b3", "C:min/b3"),
        ("C:maj/4", "C:maj/4"),
        ("C:maj/2", "C:sus2/2"),
        ("X", "N"),
    ],
)
def test_majmin_inv_nocomp(ref_label, est_label):
    __check_not_comparable(mir_eval.chord.majmin_inv, ref_label, est_label)


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("C:min", "C:min7", 0.0),
        ("C:maj", "C:maj6", 0.0),
        ("C:7", "C:9", 1.0),
        ("C:maj7", "C:maj7/5", 1.0),
        ("C:7/3", "C:11/b7", 1.0),
        ("G:min", "G:sus2", 0.0),
        ("C:maj", "G:maj", 0.0),
        ("C:7", "C:maj7", 0.0),
    ],
)
def test_sevenths(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.sevenths, ref_label, est_label, score)


@pytest.mark.parametrize(
    "ref_label, est_label", [("C:sus2", "C:sus2/2"), ("C:hdim7", "C:hdim7"), ("X", "N")]
)
def test_sevenths_nocomp(ref_label, est_label):
    __check_not_comparable(mir_eval.chord.sevenths, ref_label, est_label)


@pytest.mark.parametrize(
    "ref_label, est_label, score",
    [
        ("C:maj7/5", "C:maj7/3", 0.0),
        ("G:min", "G:min/b3", 0.0),
        ("C:7/5", "C:13/5", 1.0),
        ("C:min7/b7", "C:min7/b7", 1.0),
    ],
)
def test_sevenths_inv(ref_label, est_label, score):
    __check_one_metric(mir_eval.chord.sevenths_inv, ref_label, est_label, score)


@pytest.mark.parametrize(
    "ref_label, est_label", [("C:dim7/b3", "C:dim7/b3"), ("X", "N")]
)
def test_sevenths_inv_nocomp(ref_label, est_label):
    __check_not_comparable(mir_eval.chord.sevenths_inv, ref_label, est_label)


def test_directional_hamming_distance():
    ref_ivs = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    est_ivs = np.array([[0.0, 0.9], [0.9, 1.8], [1.8, 2.5]])
    dhd_ref_to_est = (0.1 + 0.2 + 0.5) / 3.0
    dhd_est_to_ref = (0.0 + 0.1 + 0.2) / 2.5

    dhd = mir_eval.chord.directional_hamming_distance
    assert np.allclose(dhd_ref_to_est, dhd(ref_ivs, est_ivs))
    assert np.allclose(dhd_est_to_ref, dhd(est_ivs, ref_ivs))
    assert np.allclose(0, dhd(ref_ivs, ref_ivs))
    assert np.allclose(0, dhd(est_ivs, est_ivs))

    ivs_overlap_all = np.array([[0.0, 1.0], [0.9, 2.0]])
    ivs_overlap_one = np.array([[0.0, 1.0], [0.9, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        dhd(ivs_overlap_all, est_ivs)
    with pytest.raises(ValueError):
        dhd(ivs_overlap_one, est_ivs)


def test_segmentation_functions():
    ref_ivs = np.array([[0.0, 2.0], [2.0, 2.5], [2.5, 3.2]])
    est_ivs = np.array([[0.0, 3.0], [3.0, 3.5]])
    true_oseg = 1.0 - 0.2 / 3.2
    true_useg = 1.0 - (1.0 + 0.2) / 3.5
    true_seg = min(true_oseg, true_useg)
    assert np.allclose(true_oseg, mir_eval.chord.overseg(ref_ivs, est_ivs))
    assert np.allclose(true_useg, mir_eval.chord.underseg(ref_ivs, est_ivs))
    assert np.allclose(true_seg, mir_eval.chord.seg(ref_ivs, est_ivs))

    ref_ivs = np.array([[0.0, 2.0], [2.0, 2.5], [2.5, 3.2]])
    est_ivs = np.array([[0.0, 2.0], [2.0, 2.5], [2.5, 3.2]])
    true_oseg = 1.0
    true_useg = 1.0
    true_seg = 1.0
    assert np.allclose(true_oseg, mir_eval.chord.overseg(ref_ivs, est_ivs))
    assert np.allclose(true_useg, mir_eval.chord.underseg(ref_ivs, est_ivs))
    assert np.allclose(true_seg, mir_eval.chord.seg(ref_ivs, est_ivs))

    ref_ivs = np.array([[0.0, 2.0], [2.0, 2.5], [2.5, 3.2]])
    est_ivs = np.array([[0.0, 3.2]])
    true_oseg = 1.0
    true_useg = 1 - 1.2 / 3.2
    true_seg = min(true_oseg, true_useg)
    assert np.allclose(true_oseg, mir_eval.chord.overseg(ref_ivs, est_ivs))
    assert np.allclose(true_useg, mir_eval.chord.underseg(ref_ivs, est_ivs))
    assert np.allclose(true_seg, mir_eval.chord.seg(ref_ivs, est_ivs))

    ref_ivs = np.array([[0.0, 2.0], [2.0, 2.5], [2.5, 3.2]])
    est_ivs = np.array([[3.2, 3.5]])
    true_oseg = 1.0
    true_useg = 1.0
    true_seg = 1.0
    assert np.allclose(true_oseg, mir_eval.chord.overseg(ref_ivs, est_ivs))
    assert np.allclose(true_useg, mir_eval.chord.underseg(ref_ivs, est_ivs))
    assert np.allclose(true_seg, mir_eval.chord.seg(ref_ivs, est_ivs))


def test_merge_chord_intervals():
    intervals = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3], [3.0, 4.0], [4.0, 5.0]])
    labels = ["C:maj", "C:(1,3,5)", "A:maj", "A:maj7", "A:maj7/3"]
    assert np.allclose(
        np.array([[0.0, 2.0], [2.0, 3], [3.0, 4.0], [4.0, 5.0]]),
        mir_eval.chord.merge_chord_intervals(intervals, labels),
    )


def test_weighted_accuracy():
    # First, test for a warning on empty beats
    with pytest.warns(UserWarning, match="No nonzero weights, returning 0"):
        score = mir_eval.chord.weighted_accuracy(
            np.array([1, 0, 1]), np.array([0, 0, 0])
        )
        # And that the metric is 0
        assert np.allclose(score, 0)

    # len(comparisons) must equal len(weights)
    comparisons = np.array([1, 0, 1])
    weights = np.array([1, 1])

    with pytest.raises(ValueError):
        mir_eval.chord.weighted_accuracy(comparisons, weights)

    # Weights must all be positive
    comparisons = np.array([1, 1])
    weights = np.array([-1, -1])
    with pytest.raises(ValueError):
        mir_eval.chord.weighted_accuracy(comparisons, weights)

    # Make sure accuracy = 1 and 0 when all comparisons are True and False resp
    comparisons = np.array([1, 1, 1])
    weights = np.array([1, 1, 1])
    score = mir_eval.chord.weighted_accuracy(comparisons, weights)
    assert np.allclose(score, 1)
    comparisons = np.array([0, 0, 0])
    score = mir_eval.chord.weighted_accuracy(comparisons, weights)
    assert np.allclose(score, 0)


@pytest.mark.parametrize("chord_data", file_sets, indirect=True)
def test_chord_functions(chord_data):
    ref_intervals, ref_labels, est_intervals, est_labels, expected_scores = chord_data

    # Compute scores
    scores = mir_eval.chord.evaluate(
        ref_intervals, ref_labels, est_intervals, est_labels
    )
    # Compare them
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)


def test_quality_to_bitmap():
    # Test simple case
    assert np.all(
        mir_eval.chord.quality_to_bitmap("maj")
        == np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    )


@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
@pytest.mark.parametrize("quality", ["maj5", "2", "#7"])
def test_quality_to_bitmap_fail(quality):
    # Check exceptions for qualities not in the QUALITIES list
    mir_eval.chord.quality_to_bitmap(quality)


def test_validate():
    # Test that the validate function raises the appropriate errors and
    # warnings
    with pytest.warns() as w:
        # First, test for warnings on empty labels
        mir_eval.chord.validate([], [])
    assert len(w) == 2
    assert issubclass(w[-1].category, UserWarning)
    assert str(w[-1].message) == "Estimated labels are empty"
    assert issubclass(w[-2].category, UserWarning)
    assert str(w[-2].message) == "Reference labels are empty"

    # Test that error is thrown on different-length labels
    with pytest.raises(ValueError):
        mir_eval.chord.validate([], ["C"])


@pytest.mark.parametrize("quality", list(mir_eval.chord.QUALITIES.keys()))
def test_chord_qualities(quality):
    if len(quality) > 0:
        mir_eval.chord.encode(f"C:{quality}", reduce_extended_chords=True)
    else:
        mir_eval.chord.encode("C", reduce_extended_chords=True)

# CREATED: 2/9/16 2:27 PM by Justin Salamon <justin.salamon@nyu.edu>

import mir_eval
import numpy as np
import glob
import json
import pytest

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/transcription/ref*.txt"
EST_GLOB = "data/transcription/est*.txt"
SCORES_GLOB = "data/transcription/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


@pytest.fixture
def transcription_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    # Load in an example segmentation annotation
    ref_int, ref_pitch = mir_eval.io.load_valued_intervals(ref_f)
    # Load in estimated transcription
    est_int, est_pitch = mir_eval.io.load_valued_intervals(est_f)

    return ref_int, ref_pitch, est_int, est_pitch, expected_scores


REF = np.array(
    [
        [0.100, 0.300, 220.000],
        [0.300, 0.400, 246.942],
        [0.500, 0.600, 277.183],
        [0.550, 0.650, 293.665],
    ]
)

EST = np.array(
    [
        [0.120, 0.290, 225.000],
        [0.300, 0.340, 246.942],
        [0.500, 0.600, 500.000],
        [0.550, 0.600, 293.665],
        [0.560, 0.650, 293.665],
    ]
)

SCORES = {
    "Precision": 0.4,
    "Recall": 0.5,
    "F-measure": 0.4444444444444445,
    "Average_Overlap_Ratio": 0.675,
    "Precision_no_offset": 0.6,
    "Recall_no_offset": 0.75,
    "F-measure_no_offset": 0.6666666666666665,
    "Average_Overlap_Ratio_no_offset": 0.5833333333333333,
}

ONSET_SCORES = {
    "Onset_Precision": 0.8,
    "Onset_Recall": 1.0,
    "Onset_F-measure": 0.8888888888888889,
}

OFFSET_SCORES = {
    "Offset_Precision": 0.6,
    "Offset_Recall": 0.75,
    "Offset_F-measure": 0.6666666666666665,
}


def test_match_note_offsets():
    ref_int = REF[:, :2]
    est_int = EST[:, :2]

    matching = mir_eval.transcription.match_note_offsets(ref_int, est_int)

    assert matching == [(0, 0), (2, 2), (3, 3)]


def test_match_note_offsets_strict():
    ref_int = REF[:, :2]
    est_int = EST[:, :2]

    matching = mir_eval.transcription.match_note_offsets(ref_int, est_int, strict=True)

    assert matching == [(0, 0), (2, 2), (3, 4)]


def test_match_note_onsets():
    ref_int = REF[:, :2]
    est_int = EST[:, :2]

    matching = mir_eval.transcription.match_note_onsets(ref_int, est_int)

    assert matching == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_match_note_onsets_strict():
    ref_int = REF[:, :2]
    est_int = EST[:, :2]

    matching = mir_eval.transcription.match_note_onsets(ref_int, est_int, strict=True)

    assert matching == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_match_notes():
    ref_int, ref_pitch = REF[:, :2], REF[:, 2]
    est_int, est_pitch = EST[:, :2], EST[:, 2]

    matching = mir_eval.transcription.match_notes(
        ref_int, ref_pitch, est_int, est_pitch
    )

    assert matching == [(0, 0), (3, 3)]

    matching = mir_eval.transcription.match_notes(
        ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None
    )

    assert matching == [(0, 0), (1, 1), (3, 3)]


def test_match_notes_strict():
    ref_int, ref_pitch = np.array([[0, 1]]), np.array([100])
    est_int, est_pitch = np.array([[0.05, 1]]), np.array([100])

    matching = mir_eval.transcription.match_notes(
        ref_int, ref_pitch, est_int, est_pitch, strict=True
    )

    assert matching == []


def test_precision_recall_f1_overlap():
    # load test data
    ref_int, ref_pitch = REF[:, :2], REF[:, 2]
    est_int, est_pitch = EST[:, :2], EST[:, 2]

    (
        precision,
        recall,
        f_measure,
        avg_overlap_ratio,
    ) = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch, est_int, est_pitch
    )

    scores_gen = np.array([precision, recall, f_measure, avg_overlap_ratio])
    scores_exp = np.array(
        [
            SCORES["Precision"],
            SCORES["Recall"],
            SCORES["F-measure"],
            SCORES["Average_Overlap_Ratio"],
        ]
    )
    assert np.allclose(scores_exp, scores_gen, atol=A_TOL)

    (
        precision,
        recall,
        f_measure,
        avg_overlap_ratio,
    ) = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None
    )

    scores_gen = np.array([precision, recall, f_measure, avg_overlap_ratio])
    scores_exp = np.array(
        [
            SCORES["Precision_no_offset"],
            SCORES["Recall_no_offset"],
            SCORES["F-measure_no_offset"],
            SCORES["Average_Overlap_Ratio_no_offset"],
        ]
    )
    assert np.allclose(scores_exp, scores_gen, atol=A_TOL)


def test_onset_precision_recall_f1():
    # load test data
    ref_int = REF[:, :2]
    est_int = EST[:, :2]

    precision, recall, f_measure = mir_eval.transcription.onset_precision_recall_f1(
        ref_int, est_int
    )

    scores_gen = np.array([precision, recall, f_measure])
    scores_exp = np.array(
        [
            ONSET_SCORES["Onset_Precision"],
            ONSET_SCORES["Onset_Recall"],
            ONSET_SCORES["Onset_F-measure"],
        ]
    )
    assert np.allclose(scores_exp, scores_gen, atol=A_TOL)


def test_offset_precision_recall_f1():
    # load test data
    ref_int = REF[:, :2]
    est_int = EST[:, :2]

    precision, recall, f_measure = mir_eval.transcription.offset_precision_recall_f1(
        ref_int, est_int
    )

    scores_gen = np.array([precision, recall, f_measure])
    scores_exp = np.array(
        [
            OFFSET_SCORES["Offset_Precision"],
            OFFSET_SCORES["Offset_Recall"],
            OFFSET_SCORES["Offset_F-measure"],
        ]
    )
    assert np.allclose(scores_exp, scores_gen, atol=A_TOL)


@pytest.mark.parametrize("transcription_data", file_sets, indirect=True)
def test_regression(transcription_data):
    ref_int, ref_pitch, est_int, est_pitch, expected_scores = transcription_data

    scores = mir_eval.transcription.evaluate(ref_int, ref_pitch, est_int, est_pitch)
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "ref_pitch, est_pitch",
    [(np.array([-100]), np.array([100])), (np.array([100]), np.array([-100]))],
)
def test_invalid_pitch(ref_pitch, est_pitch):
    ref_int = np.array([[0, 1]])
    mir_eval.transcription.validate(ref_int, ref_pitch, ref_int, est_pitch)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "ref_int, est_int",
    [
        (np.array([[0, 1], [2, 3]]), np.array([[0, 1]])),
        (np.array([[0, 1]]), np.array([[0, 1], [2, 3]])),
    ],
)
def test_inconsistent_int_pitch(ref_int, est_int):
    ref_pitch = np.array([100])
    mir_eval.transcription.validate(ref_int, ref_pitch, est_int, ref_pitch)


def test_empty_ref():
    ref_int, ref_pitch = np.empty(shape=(0, 2)), np.array([])
    est_int, est_pitch = np.array([[0, 1]]), np.array([100])

    with pytest.warns(UserWarning, match="Reference notes are empty"):
        mir_eval.transcription.validate(ref_int, ref_pitch, est_int, est_pitch)


def test_empty_est():
    ref_int, ref_pitch = np.array([[0, 1]]), np.array([100])
    est_int, est_pitch = np.empty(shape=(0, 2)), np.array([])

    with pytest.warns(UserWarning, match="Estimated notes are empty"):
        mir_eval.transcription.validate(ref_int, ref_pitch, est_int, est_pitch)


@pytest.mark.filterwarnings("ignore:.*notes are empty")
def test_precision_recall_f1_overlap_empty():
    ref_int, ref_pitch = np.empty(shape=(0, 2)), np.array([])
    est_int, est_pitch = np.array([[0, 1]]), np.array([100])

    (
        precision,
        recall,
        f1,
        avg_overlap_ratio,
    ) = mir_eval.transcription.precision_recall_f1_overlap(
        ref_int, ref_pitch, est_int, est_pitch
    )

    assert (precision, recall, f1) == (0, 0, 0)

    (
        precision,
        recall,
        f1,
        avg_overlap_ratio,
    ) = mir_eval.transcription.precision_recall_f1_overlap(
        est_int, est_pitch, ref_int, ref_pitch
    )

    assert (precision, recall, f1) == (0, 0, 0)


@pytest.mark.filterwarnings("ignore:.*notes are empty")
def test_onset_precision_recall_f1_empty():
    ref_int = np.empty(shape=(0, 2))
    est_int = np.array([[0, 1]])

    precision, recall, f1 = mir_eval.transcription.onset_precision_recall_f1(
        ref_int, est_int
    )

    assert (precision, recall, f1) == (0, 0, 0)

    precision, recall, f1 = mir_eval.transcription.onset_precision_recall_f1(
        est_int, ref_int
    )

    assert (precision, recall, f1) == (0, 0, 0)


@pytest.mark.filterwarnings("ignore:.*notes are empty")
def test_offset_precision_recall_f1_empty():
    ref_int = np.empty(shape=(0, 2))
    est_int = np.array([[0, 1]])

    precision, recall, f1 = mir_eval.transcription.offset_precision_recall_f1(
        ref_int, est_int
    )

    assert (precision, recall, f1) == (0, 0, 0)

    precision, recall, f1 = mir_eval.transcription.offset_precision_recall_f1(
        est_int, ref_int
    )

    assert (precision, recall, f1) == (0, 0, 0)

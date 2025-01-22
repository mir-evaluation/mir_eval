import pytest
import mir_eval
import numpy as np
import glob
import json

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/transcription_velocity/ref*.txt"
EST_GLOB = "data/transcription_velocity/est*.txt"
SCORES_GLOB = "data/transcription_velocity/output*.json"
ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))


def _load_transcription_velocity(filename):
    """Loader for data in the format start, end, pitch, velocity."""
    starts, ends, pitches, velocities = mir_eval.io.load_delimited(
        filename, [float, float, int, int]
    )
    # Stack into an interval matrix
    intervals = np.array([starts, ends]).T
    # return pitches and velocities as np.ndarray
    pitches = np.array(pitches)
    velocities = np.array(velocities)
    return intervals, pitches, velocities


@pytest.fixture
def velocity_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_scores = json.load(f)
    # Load in reference transcription
    ref_int, ref_pitch, ref_vel = _load_transcription_velocity(ref_f)
    # Load in estimated transcription
    est_int, est_pitch, est_vel = _load_transcription_velocity(est_f)

    return (ref_int, ref_pitch, ref_vel), (est_int, est_pitch, est_vel), expected_scores


def test_negative_velocity():
    good_i, good_p, good_v = np.array([[0, 1]]), np.array([100]), np.array([1])
    bad_i, bad_p, bad_v = np.array([[0, 1]]), np.array([100]), np.array([-1])

    with pytest.raises(ValueError):
        mir_eval.transcription_velocity.validate(
            bad_i, bad_p, bad_v, good_i, good_p, good_v
        )
    with pytest.raises(ValueError):
        mir_eval.transcription_velocity.validate(
            good_i, good_p, good_v, bad_i, bad_p, bad_v
        )


def test_wrong_shape_velocity():
    good_i, good_p, good_v = np.array([[0, 1]]), np.array([100]), np.array([1])
    bad_i, bad_p, bad_v = np.array([[0, 1]]), np.array([100]), np.array([1, 2])

    with pytest.raises(ValueError):
        mir_eval.transcription_velocity.validate(
            bad_i, bad_p, bad_v, good_i, good_p, good_v
        )
    with pytest.raises(ValueError):
        mir_eval.transcription_velocity.validate(
            good_i, good_p, good_v, bad_i, bad_p, bad_v
        )


def test_precision_recall_f1_overlap():
    # Simple unit test
    ref_i = np.array([[0, 1], [0.5, 0.7], [1, 2]])
    ref_p = np.array([100, 110, 80])
    ref_v = np.array([10, 90, 110])
    est_i = np.array([[0, 1], [0.5, 0.7], [1, 2]])
    est_p = np.array([100, 110, 80])
    est_v = np.array([10, 70, 110])
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_i, ref_p, ref_v, est_i, est_p, est_v
    )
    assert np.allclose((p, r, f, o), (2 / 3.0, 2 / 3.0, 2 / 3.0, 1.0))
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_i, ref_p, ref_v, est_i, est_p, est_v, velocity_tolerance=0.3
    )
    assert np.allclose((p, r, f, o), (1.0, 1.0, 1.0, 1.0))


# Suppressing this warning. We know the notes are empty, that's not the point.
@pytest.mark.filterwarnings("ignore:.*notes are empty")
def test_precision_recall_f1_overlap_empty():
    good_i, good_p, good_v = np.array([[0, 1]]), np.array([100]), np.array([1])
    bad_i, bad_p, bad_v = np.empty((0, 2)), np.array([]), np.array([])
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        good_i, good_p, good_v, bad_i, bad_p, bad_v
    )
    assert (p, r, f, o) == (0.0, 0.0, 0.0, 0.0)
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        bad_i, bad_p, bad_v, good_i, good_p, good_v
    )
    assert (p, r, f, o) == (0.0, 0.0, 0.0, 0.0)


def test_precision_recall_f1_overlap_no_overlap():
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        np.array([[1, 2]]),
        np.array([1]),
        np.array([1]),
        np.array([[3, 4]]),
        np.array([1]),
        np.array([1]),
    )
    assert (p, r, f, o) == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("velocity_data", file_sets, indirect=True)
def test_regression(velocity_data):
    (
        (ref_int, ref_pitch, ref_vel),
        (est_int, est_pitch, est_vel),
        expected_scores,
    ) = velocity_data

    scores = mir_eval.transcription_velocity.evaluate(
        ref_int, ref_pitch, ref_vel, est_int, est_pitch, est_vel
    )
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL), metric

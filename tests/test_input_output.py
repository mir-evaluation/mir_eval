"""Unit tests for input/output functions"""

import numpy as np
import json
import mir_eval
import warnings
import pathlib
import tempfile
import pytest


def test_load_delimited():
    # Test for ValueError when a non-string or file handle is passed
    with pytest.raises(IOError):
        mir_eval.io.load_delimited(None, [int])

    # Test for a value error when the wrong number of columns is passed
    with tempfile.TemporaryFile("r+") as f:
        f.write("10 20")
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_delimited(f, [int, int, int])

    # Test for a value error on conversion failure
    with tempfile.TemporaryFile("r+") as f:
        f.write("10 a 30")
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_delimited(f, [int, int, int])


def test_load_delimited_commented():
    with tempfile.TemporaryFile("r+") as f:
        f.write("; some comment\n10 20\n30 50")
        f.seek(0)
        col1, col2 = mir_eval.io.load_delimited(f, [int, int], comment=";")
        assert np.allclose(col1, [10, 30])
        assert np.allclose(col2, [20, 50])

        # Rewind and try with the default comment character
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_delimited(f, [int, int])

        # Rewind and try with no comment support
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_delimited(f, [int, int], comment=None)


def test_load_delimited_nocomment():
    with tempfile.TemporaryFile("r+") as f:
        f.write("10 20\n30 50")
        f.seek(0)
        col1, col2 = mir_eval.io.load_delimited(f, [int, int])
        assert np.allclose(col1, [10, 30])
        assert np.allclose(col2, [20, 50])

        # Rewind and try with a different comment char
        f.seek(0)
        col1, col2 = mir_eval.io.load_delimited(f, [int, int], comment=";")
        assert np.allclose(col1, [10, 30])
        assert np.allclose(col2, [20, 50])

        # Rewind and try with no different comment string
        f.seek(0)
        col1, col2 = mir_eval.io.load_delimited(f, [int, int], comment=None)
        assert np.allclose(col1, [10, 30])
        assert np.allclose(col2, [20, 50])


@pytest.mark.parametrize(
    "file_or_path", ["data/beat/ref01.txt", pathlib.Path("data/beat/ref01.txt")]
)
def test_load_delimited_file_or_path(file_or_path):
    # Tests the load_delimited routine with either a string or a
    # pathlib.Path object as the first argument
    data = mir_eval.io.load_delimited(file_or_path, [float])
    assert len(data) == 635


def test_load_events():
    # Test for a warning when invalid events are supplied
    with tempfile.TemporaryFile("r+") as f:
        with pytest.warns(UserWarning, match="Events should be in increasing order."):
            # Non-increasing is invalid
            f.write("10\n9")
            f.seek(0)
            events = mir_eval.io.load_events(f)
            # Make sure events were read in correctly
            assert np.all(events == [10, 9])


def test_load_labeled_events():
    # Test for a value error when invalid labeled events are supplied
    with tempfile.TemporaryFile("r+") as f:
        with pytest.warns(UserWarning, match="Events should be in increasing order."):
            # Non-increasing is invalid
            f.write("10 a\n9 b")
            f.seek(0)
            events, labels = mir_eval.io.load_labeled_events(f)
            # Make sure events were read in correctly
            assert np.all(events == [10, 9])
            # Make sure labels were read in correctly
            assert labels == ["a", "b"]


def test_load_intervals():
    # Test for a value error when invalid labeled events are supplied
    with tempfile.TemporaryFile("r+") as f:
        with pytest.warns(
            UserWarning, match="All interval durations must be strictly positive"
        ):
            # Non-increasing is invalid
            f.write("10 9\n9 10")
            f.seek(0)
            intervals = mir_eval.io.load_intervals(f)
            # Make sure intervals were read in correctly
            assert np.all(intervals == [[10, 9], [9, 10]])


def test_load_labeled_intervals():
    # Test for a value error when invalid labeled events are supplied
    with tempfile.TemporaryFile("r+") as f:
        with pytest.warns(
            UserWarning, match="All interval durations must be strictly positive"
        ):
            # Non-increasing is invalid
            f.write("10 9 a\n9 10 b")
            f.seek(0)
            intervals, labels = mir_eval.io.load_labeled_intervals(f)
            # Make sure intervals were read in correctly
            assert np.all(intervals == [[10, 9], [9, 10]])
            assert labels == ["a", "b"]


def test_load_valued_intervals():
    # Test for a value error when invalid valued events are supplied
    with tempfile.TemporaryFile("r+") as f:
        with pytest.warns(
            UserWarning, match="All interval durations must be strictly positive"
        ):
            # Non-increasing is invalid
            f.write("10 9 5\n9 10 6")
            f.seek(0)
            intervals, values = mir_eval.io.load_valued_intervals(f)
            # Make sure intervals were read in correctly
            assert np.all(intervals == [[10, 9], [9, 10]])
            assert np.all(values == [5, 6])


def test_load_ragged_time_series():
    # Test for ValueError when a non-string or file handle is passed
    with pytest.raises(IOError):
        mir_eval.io.load_ragged_time_series(None, float, header=False)
    # Test for a value error on conversion failure
    with tempfile.TemporaryFile("r+") as f:
        f.write("10 a 30")
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_ragged_time_series(f, float, header=False)
    # Test for a value error on invalid time stamp
    with tempfile.TemporaryFile("r+") as f:
        f.write("a 10 30")
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_ragged_time_series(f, int, header=False)
    # Test for a value error on invalid time stamp with header
    with tempfile.TemporaryFile("r+") as f:
        f.write("x y z\na 10 30")
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_ragged_time_series(f, int, header=True)

    with tempfile.TemporaryFile("r+") as f:
        f.write("#comment\n0 1 2\n3 4\n# comment\n5 6 7")
        f.seek(0)
        times, values = mir_eval.io.load_ragged_time_series(
            f, int, header=False, comment="#"
        )
        assert np.allclose(times, [0, 3, 5])
        assert np.allclose(values[0], [1, 2])
        assert np.allclose(values[1], [4])
        assert np.allclose(values[2], [6, 7])

        # Rewind with a wrong comment string
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_ragged_time_series(f, int, header=False, comment="%")

        # Rewind with no comment string
        f.seek(0)
        with pytest.raises(ValueError):
            mir_eval.io.load_ragged_time_series(f, int, header=False, comment=None)


def test_load_tempo():
    # Test the tempo loader
    tempi, weight = mir_eval.io.load_tempo("data/tempo/ref01.lab")

    assert np.allclose(tempi, [60, 120])
    assert weight == 0.5


@pytest.mark.xfail(raises=ValueError)
def test_load_tempo_multiline():
    tempi, weight = mir_eval.io.load_tempo("data/tempo/bad00.lab")


@pytest.mark.xfail(raises=ValueError)
def test_load_tempo_badweight():
    tempi, weight = mir_eval.io.load_tempo("data/tempo/bad01.lab")


def test_load_bad_tempi():
    with pytest.warns(UserWarning, match="non-negative numbers"):
        tempi, weight = mir_eval.io.load_tempo("data/tempo/bad02.lab")

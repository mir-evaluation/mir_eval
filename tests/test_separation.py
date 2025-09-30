"""
unit tests for mir_eval.separation

load randomly generated source and estimated source signals and
the output from BSS_eval MATLAB implementation, make sure the results
from mir_eval numerically match.
"""

import numpy as np
import mir_eval
import glob
import pytest
import json
import os

A_TOL = 1e-2

REF_GLOB = "data/separation/ref*"
EST_GLOB = "data/separation/est*"
SCORES_GLOB = "data/separation/output*.json"

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0
file_sets = list(zip(ref_files, est_files, sco_files))

# Skip separation tests since deprecation
pytest.skip(allow_module_level=True)


@pytest.fixture
def separation_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f) as f:
        expected_results = json.load(f)
        expected_sources = expected_results["Sources"]
        expected_frames = expected_results["Framewise"]
        expected_images = expected_results["Images"]
        expected_image_frames = expected_results["Images Framewise"]

    # Load in example source separation data
    ref_sources = __load_and_stack_wavs(ref_f)
    est_sources = __load_and_stack_wavs(est_f)

    # Test inference for single source passed as single dimensional array
    if ref_sources.shape[0] == 1 and est_sources.shape[0] == 1:
        ref_sources = ref_sources[0]
        est_sources = est_sources[0]

    return (
        ref_sources,
        est_sources,
        expected_sources,
        expected_frames,
        expected_images,
        expected_image_frames,
    )


@pytest.fixture(autouse=True)
def seed_rng():
    # Seed the RNG before each test run
    np.random.seed(1999)


def __load_and_stack_wavs(directory):
    """Load all wavs in a directory and stack them vertically into a matrix"""
    stacked_audio_data = []
    global_fs = None
    for f in sorted(glob.glob(os.path.join(directory, "*.wav"))):
        audio_data, fs = mir_eval.io.load_wav(f)
        assert global_fs is None or fs == global_fs
        global_fs = fs
        stacked_audio_data.append(audio_data)
    return np.vstack(stacked_audio_data)


def __generate_multichannel(mono_sig, nchan=2, gain=1.0, reverse=False):
    """Turn a single channel (ie. mono) audio sample into a multichannel
    (e.g. stereo)
    Note: to achieve channels of silence pass gain=0
    """
    # add the channels dimension
    input_3d = np.atleast_3d(mono_sig)
    # get the desired number of channels
    stackin = [input_3d] * nchan
    # apply the gain to the new channels
    stackin[1:] = np.multiply(gain, stackin[1:])
    if reverse:
        # reverse the new channels
        stackin[1:] = stackin[1:][:][::-1]
    return np.dstack(stackin)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.separation.bss_eval_sources,
        mir_eval.separation.bss_eval_sources_framewise,
        mir_eval.separation.bss_eval_images,
        mir_eval.separation.bss_eval_images_framewise,
    ],
)
def test_empty_input(metric):
    if (
        metric == mir_eval.separation.bss_eval_sources
        or metric == mir_eval.separation.bss_eval_images
    ):
        args = [np.array([]), np.array([])]
    elif (
        metric == mir_eval.separation.bss_eval_sources_framewise
        or metric == mir_eval.separation.bss_eval_images_framewise
    ):
        args = [np.array([]), np.array([]), 40, 20]

    with pytest.warns(UserWarning, match="is empty") as record:

        # First, test for a warning on empty audio data
        metric(*args)
        # And that the metric returns empty arrays
        assert np.allclose(metric(*args), np.array([]))

        # These warning counters are now offset by 1 because of the deprecation message
        assert "reference_sources is empty" in str(record[1].message)
        assert "estimated_sources is empty" in str(record[2].message)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.separation.bss_eval_sources,
        mir_eval.separation.bss_eval_sources_framewise,
        mir_eval.separation.bss_eval_images,
        mir_eval.separation.bss_eval_images_framewise,
    ],
)
def test_silent_input(metric):
    # Test for error when there is a silent reference/estimated source
    if (
        metric == mir_eval.separation.bss_eval_images
        or metric == mir_eval.separation.bss_eval_images_framewise
    ):
        ref_sources = np.vstack(
            (np.zeros((1, 100, 2)), np.random.random_sample((2, 100, 2)))
        )
        est_sources = np.vstack(
            (np.zeros((1, 100, 2)), np.random.random_sample((2, 100, 2)))
        )
    else:
        ref_sources = np.vstack((np.zeros(100), np.random.random_sample((2, 100))))
        est_sources = np.vstack((np.zeros(100), np.random.random_sample((2, 100))))
    if (
        metric == mir_eval.separation.bss_eval_sources
        or metric == mir_eval.separation.bss_eval_images
    ):
        with pytest.raises(ValueError):
            metric(ref_sources[:2], est_sources[1:])
        with pytest.raises(ValueError):
            metric(ref_sources[1:], est_sources[:2])
    elif (
        metric == mir_eval.separation.bss_eval_sources_framewise
        or metric == mir_eval.separation.bss_eval_images_framewise
    ):
        with pytest.raises(ValueError):
            metric(ref_sources[:2], est_sources[1:], 40, 20)
        with pytest.raises(ValueError):
            metric(ref_sources[1:], est_sources[:2], 40, 20)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.separation.bss_eval_sources_framewise,
        mir_eval.separation.bss_eval_images_framewise,
    ],
)
def test_partial_silence(metric):
    # Test for a full window of silence in reference/estimated source
    if metric == mir_eval.separation.bss_eval_sources_framewise:
        silence = np.zeros((2, 20))
        sound = np.random.random_sample((2, 20))
    elif metric == mir_eval.separation.bss_eval_images_framewise:
        silence = np.zeros((2, 20, 2))
        sound = np.random.random_sample((2, 20, 2))
    # test with silence in the reference
    results = metric(
        np.concatenate((sound, silence, sound), axis=1),
        np.concatenate((sound, sound, sound), axis=1),
        window=10,
        hop=10,
    )
    for measure in results:
        for idx, source in enumerate(measure):
            if idx < 2 or idx > 3:
                assert not np.isnan(source[idx])
            elif idx < 4:
                assert np.isnan(source[idx])
            else:
                raise ValueError("Testing error in partial silence test")
    # test with silence in the estimate
    results = metric(
        np.concatenate((sound, sound, sound), axis=1),
        np.concatenate((sound, silence, sound), axis=1),
        window=10,
        hop=10,
    )
    for measure in results:
        for idx, source in enumerate(measure):
            if idx < 2 or idx > 3:
                assert not np.isnan(source[idx])
            elif idx < 4:
                assert np.isnan(source[idx])
            else:
                raise ValueError("Testing error in partial silence test")


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.separation.bss_eval_sources,
        mir_eval.separation.bss_eval_sources_framewise,
        mir_eval.separation.bss_eval_images,
        mir_eval.separation.bss_eval_images_framewise,
    ],
)
def test_incompatible_shapes(metric):
    # Test for error when shape is different
    if (
        metric == mir_eval.separation.bss_eval_images
        or metric == mir_eval.separation.bss_eval_images_framewise
    ):
        sources_4 = np.random.random_sample((4, 100, 2))
        sources_3 = np.random.random_sample((3, 100, 2))
        sources_4_chan = np.random.random_sample((4, 100, 3))
    else:
        sources_4 = np.random.random_sample((4, 100))
        sources_3 = np.random.random_sample((3, 100))
    if (
        metric == mir_eval.separation.bss_eval_sources
        or metric == mir_eval.separation.bss_eval_images
    ):
        args1 = [sources_3, sources_4]
        args2 = [sources_4, sources_3]
    elif (
        metric == mir_eval.separation.bss_eval_sources_framewise
        or metric == mir_eval.separation.bss_eval_images_framewise
    ):
        args1 = [sources_3, sources_4, 40, 20]
        args2 = [sources_4, sources_3, 40, 20]
    with pytest.raises(ValueError):
        metric(*args1)
    with pytest.raises(ValueError):
        metric(*args2)
    if (
        metric == mir_eval.separation.bss_eval_images
        or metric == mir_eval.separation.bss_eval_images_framewise
    ):
        with pytest.raises(ValueError):
            metric(sources_4, sources_4_chan)


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.separation.bss_eval_sources,
        mir_eval.separation.bss_eval_sources_framewise,
        mir_eval.separation.bss_eval_images,
        mir_eval.separation.bss_eval_images_framewise,
    ],
)
def test_too_many_sources(metric):
    # Test for error when too many sources or references are provided
    many_sources = np.random.random_sample((mir_eval.separation.MAX_SOURCES * 2, 400))
    if metric == mir_eval.separation.bss_eval_sources:
        with pytest.raises(ValueError):
            metric(many_sources, many_sources)
    elif metric == mir_eval.separation.bss_eval_sources_framewise:
        with pytest.raises(ValueError):
            metric(many_sources, many_sources, 40, 20)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.separation.bss_eval_sources,
        mir_eval.separation.bss_eval_sources_framewise,
        mir_eval.separation.bss_eval_images,
        mir_eval.separation.bss_eval_images_framewise,
    ],
)
def test_too_many_dimensions(metric):
    # Test for detection of too high dimensioned images
    ref_sources = np.random.random_sample((4, 100, 2, 3))
    est_sources = np.random.random_sample((4, 100, 2, 3))
    metric(ref_sources, est_sources)


@pytest.mark.parametrize(
    "metric",
    [mir_eval.separation.bss_eval_sources, mir_eval.separation.bss_eval_images],
)
def test_default_permutation(metric):
    # Test for default permutation matrix when not computing permutation
    if metric == mir_eval.separation.bss_eval_sources:
        ref_sources = np.random.random_sample((4, 100))
        est_sources = np.random.random_sample((4, 100))
    elif metric == mir_eval.separation.bss_eval_images:
        ref_sources = np.random.random_sample((4, 100, 2))
        est_sources = np.random.random_sample((4, 100, 2))
    results = metric(ref_sources, est_sources, compute_permutation=False)
    assert np.array_equal(results[-1], np.asarray([0, 1, 2, 3]))


@pytest.mark.parametrize(
    "metric",
    [
        mir_eval.separation.bss_eval_sources_framewise,
        mir_eval.separation.bss_eval_images_framewise,
    ],
)
def test_framewise_small_window(metric):
    # Test for invalid win/hop parameter detection
    if metric == mir_eval.separation.bss_eval_sources_framewise:
        ref_sources = np.random.random_sample((4, 100))
        est_sources = np.random.random_sample((4, 100))
        comparison_fcn = mir_eval.separation.bss_eval_sources
    elif metric == mir_eval.separation.bss_eval_images_framewise:
        ref_sources = np.random.random_sample((4, 100, 2))
        est_sources = np.random.random_sample((4, 100, 2))
        comparison_fcn = mir_eval.separation.bss_eval_images

    # Test with window larger than source length
    assert np.allclose(
        np.squeeze(metric(ref_sources, est_sources, window=120, hop=20)),
        comparison_fcn(ref_sources, est_sources, False),
        atol=A_TOL,
    )
    # Test with hop larger than source length
    assert np.allclose(
        np.squeeze(metric(ref_sources, est_sources, window=20, hop=120)),
        comparison_fcn(ref_sources, est_sources, False),
        atol=A_TOL,
    )


@pytest.mark.parametrize("separation_data", file_sets, indirect=True)
def test_separation_functions(separation_data):
    (
        ref_sources,
        est_sources,
        expected_sources,
        expected_frames,
        expected_images,
        expected_image_frames,
    ) = separation_data

    # Compute scores
    scores = mir_eval.separation.evaluate(
        ref_sources,
        est_sources,
        window=expected_frames["win"],
        hop=expected_frames["hop"],
    )

    # Compare them
    for key in scores:
        if "Sources - " in key:
            test_data_name = key.replace("Sources - ", "")
            assert np.allclose(
                scores[key], expected_sources[test_data_name], atol=A_TOL
            )
        elif "Sources Frames - " in key:
            test_data_name = key.replace("Sources Frames - ", "")
            assert np.allclose(scores[key], expected_frames[test_data_name], atol=A_TOL)


@pytest.mark.parametrize("separation_data", file_sets, indirect=True)
def test_separation_images(separation_data):
    (
        ref_sources,
        est_sources,
        expected_sources,
        expected_frames,
        expected_images,
        expected_image_frames,
    ) = separation_data
    # Compute scores with images
    ref_images = __generate_multichannel(ref_sources, expected_images["nchan"])
    est_images = __generate_multichannel(
        est_sources,
        expected_images["nchan"],
        expected_images["gain"],
        expected_images["reverse"],
    )
    image_scores = mir_eval.separation.evaluate(ref_images, est_images)
    # Compare them
    for key in image_scores:
        if "Images - " in key:
            test_data_name = key.replace("Images - ", "")
            assert np.allclose(
                image_scores[key], expected_images[test_data_name], atol=A_TOL
            )


@pytest.mark.parametrize("separation_data", file_sets, indirect=True)
def test_separation_images_framewise(separation_data):
    (
        ref_sources,
        est_sources,
        expected_sources,
        expected_frames,
        expected_images,
        expected_image_frames,
    ) = separation_data

    # Compute scores with images framewise
    ref_images = __generate_multichannel(ref_sources, expected_image_frames["nchan"])
    est_images = __generate_multichannel(
        est_sources,
        expected_image_frames["nchan"],
        expected_image_frames["gain"],
        expected_image_frames["reverse"],
    )
    imageframe_scores = mir_eval.separation.evaluate(
        ref_images,
        est_images,
        window=expected_image_frames["win"],
        hop=expected_image_frames["hop"],
    )
    # Compare them
    for key in imageframe_scores:
        if "Images Frames - " in key:
            test_data_name = key.replace("Images Frames - ", "")
            assert np.allclose(
                imageframe_scores[key],
                expected_image_frames[test_data_name],
                atol=A_TOL,
            )

    # Catch a few exceptions in the evaluate function
    image_scores = mir_eval.separation.evaluate(ref_images, est_images)
    # make sure sources is not being evaluated on images
    assert "Sources - Source to Distortion" not in image_scores

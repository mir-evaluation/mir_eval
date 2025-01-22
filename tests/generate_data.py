#!/usr/bin/env python
"""
Generate data for regression tests.
This is a pretty specialized file and should probably only be used if you know
what you're doing.
It expects the following directory structure for data:
    Each task has its own folder in the data directory.
    In each task folder, there are a bunch of file pairs for annotations.
    The reference annotation is ref*.txt, estimated is est*.txt.
    So, e.g., we expect to find data/beat/ref00.txt and data/beat/est00.txt.
    The resulting scores dict from the corresponding task will be written to
    output*.json.
    So, from the example above it would be written to data/beat/output00.json

To use this script, run it as a command-line program:
    ./generate_data.py task1 task2...
task1, task2, etc. are the tasks you'd like to generate data for.
So, for example, if you'd like to generate data for onset and melody,run
    ./generate_data.py onset melody
"""


import mir_eval
import glob
import json
import numpy as np
import os
import sys


def load_separation_data(folder):
    """
    Loads in a stacked matrix of the .wavs in the provided folder.
    We need this because there's no specialized loader in .io for it.
    """
    data = []
    global_fs = None
    # Load in each reference file in the supplied dir
    for reference_file in glob.glob(os.path.join(folder, "*.wav")):
        audio_data, fs = mir_eval.io.load_wav(reference_file)
        # Make sure fs is the same for all files
        assert global_fs is None or fs == global_fs
        global_fs = fs
        data.append(audio_data)
    return np.vstack(data)


def load_transcription_velocity(filename):
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


if __name__ == "__main__":
    # This dict will contain tuples of (submodule, loader, glob path)
    # The keys are 'beat', 'chord', etc.
    # Whatever is passed in as argv will be grabbed from it and the data for
    # that task will be generated.
    tasks = {}
    tasks["beat"] = (mir_eval.beat, mir_eval.io.load_events, "data/beat/{}*.txt")
    tasks["chord"] = (
        mir_eval.chord,
        mir_eval.io.load_labeled_intervals,
        "data/chord/{}*.lab",
    )
    tasks["melody"] = (
        mir_eval.melody,
        mir_eval.io.load_time_series,
        "data/melody/{}*.txt",
    )
    tasks["multipitch"] = (
        mir_eval.multipitch,
        mir_eval.io.load_ragged_time_series,
        "data/multipitch/()*.txt",
    )
    tasks["onset"] = (mir_eval.onset, mir_eval.io.load_events, "data/onset/{}*.txt")
    tasks["pattern"] = (
        mir_eval.pattern,
        mir_eval.io.load_patterns,
        "data/pattern/{}*.txt",
    )
    tasks["segment"] = (
        mir_eval.segment,
        mir_eval.io.load_labeled_intervals,
        "data/segment/{}*.lab",
    )
    tasks["separation"] = (
        mir_eval.separation,
        load_separation_data,
        "data/separation/{}*",
    )
    tasks["transcription"] = (
        mir_eval.transcription,
        mir_eval.io.load_valued_intervals,
        "data/transcription/{}*.txt",
    )
    tasks["transcription_velocity"] = (
        mir_eval.transcription_velocity,
        load_transcription_velocity,
        "data/transcription_velocity/{}*.txt",
    )
    tasks["key"] = (mir_eval.key, mir_eval.io.load_key, "data/key/{}*.txt")
    # Get task keys from argv
    for task in sys.argv[1:]:
        print(f"Generating data for {task}")
        submodule, loader, data_glob = tasks[task]
        ref_files = sorted(glob.glob(data_glob.format("ref")))
        est_files = sorted(glob.glob(data_glob.format("est")))
        # Cycle through annotation file pairs
        for ref_file, est_file in zip(ref_files, est_files):
            # Use the loader to load in data
            ref_data = loader(ref_file)
            est_data = loader(est_file)
            # Some loaders return tuples, others don't
            if type(ref_data) == tuple:
                scores = submodule.evaluate(*(ref_data + est_data))
            else:
                scores = submodule.evaluate(ref_data, est_data)
            # Write out the resulting scores dict
            output_file = ref_file.replace("ref", "output")
            output_file = os.path.splitext(output_file)[0] + ".json"
            with open(output_file, "w") as f:
                json.dump(scores, f)

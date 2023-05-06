import os
import numpy as np
import pretty_midi
import pypianoroll
from typing import List
from tqdm import tqdm


def midi_to_pianoroll(midi_path: str, resolution: int = 24) -> np.ndarray:
    """
    Convert a MIDI file to a piano roll representation.

    Args:
        midi_path (str): Path to the input MIDI file.
        resolution (int): Time resolution of the piano roll.

    Returns:
        np.ndarray: Piano roll representation of the input MIDI file.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    multitrack = pypianoroll.from_pretty_midi(midi_data, resolution=resolution)
    pianoroll = np.zeros((multitrack.tracks[0].pianoroll.shape[0], 128))

    for track in multitrack.tracks:
        pianoroll = np.maximum(pianoroll, track.pianoroll)

    return pianoroll


def process_files(midi_files: List[str], output_dir: str) -> None:
    """
    Process all MIDI files in the input list and save the preprocessed
    piano rolls in the output directory.

    Args:
        midi_files (List[str]): List of paths to MIDI files.
        output_dir (str): Path to the output directory to save the preprocessed piano rolls.
    """
    for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
        try:
            input_path = midi_file
            output_path = os.path.join(output_dir, os.path.basename(
                midi_file).replace('.mid', '.npy'))
            piano_roll = midi_to_pianoroll(input_path)
            np.save(output_path, piano_roll)

            # Check if the saved file can be loaded successfully
            _ = np.load(output_path, allow_pickle=True)
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")


def find_midi_files(input_dir: str) -> List[str]:
    """
    Get a list of all MIDI files in the input directory.

    Args:
        input_dir (str): Path to the input directory containing MIDI files.

    Returns:
        List[str]: List of paths to MIDI files in the input directory.
    """
    midi_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    return midi_files


input_dir = '../data/raw_data'
output_dir = '../data/piano_rolls'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all MIDI files in the input directory
midi_files = find_midi_files(input_dir)

# Limit the number of songs to process
num_songs_to_process = 100
midi_files = midi_files[:num_songs_to_process]

# Process all MIDI files
process_files(midi_files, output_dir)

print("Preprocessing complete!")

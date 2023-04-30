import os
import pretty_midi
import numpy as np
from tqdm import tqdm

# Set the directory where MIDI files are located
midi_dir = "data/pop"

# Set the root directory of the project
root_dir = os.path.dirname(os.path.abspath(__file__))

# Set the directory to save the piano roll data
piano_roll_dir = "data/piano_rolls"

# Set the time steps for the piano roll
time_steps_per_second = 8
time_steps_per_bar = 16
time_steps_per_beat = 4
time_steps_per_sixteenth_note = 1

# Create piano_roll_dir directory if it does not exist
os.makedirs(os.path.join(root_dir, piano_roll_dir), exist_ok=True)

# Count number of MIDI files to process
num_files = sum([len(files) for r, d, files in os.walk(midi_dir)])

# Initialize progress bar
with tqdm(total=num_files, desc="Processing MIDI files") as pbar:
    # Loop through all MIDI files in the directory
    for root, dirs, files in os.walk(midi_dir):
        for file in files:
            # Check if the file is a MIDI file
            if not file.endswith('.mid'):
                continue

            # Load the MIDI file with pretty_midi
            midi_path = os.path.join(root, file)
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_path)
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")
                continue

            # Create a piano roll from the MIDI data
            piano_roll = midi_data.get_piano_roll(
                fs=time_steps_per_second * time_steps_per_bar / time_steps_per_beat,
                times=np.arange(0, midi_data.get_end_time(),
                                1 / time_steps_per_second)
            )
            piano_roll = piano_roll.T.astype(np.uint8)

            # Save the piano roll as a numpy array at the root directory of the project
            np.save(os.path.join(root_dir, piano_roll_dir,
                    file.replace(".mid", ".npy")), piano_roll)

            # Update progress bar
            pbar.update(1)

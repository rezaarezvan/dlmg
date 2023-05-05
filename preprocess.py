import os
import pretty_midi
import numpy as np
from tqdm import tqdm


def process_midi_file(midi_path, piano_roll_dir, time_steps_per_second, time_steps_per_bar, time_steps_per_beat):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return

    piano_roll = midi_data.get_piano_roll(
        fs=time_steps_per_second * time_steps_per_bar / time_steps_per_beat,
        times=np.arange(0, midi_data.get_end_time(),
                        1 / time_steps_per_second)
    )
    piano_roll = piano_roll.T.astype(np.uint8)

    output_file = os.path.join(piano_roll_dir, os.path.basename(
        midi_path).replace(".mid", ".npy"))
    np.save(output_file, piano_roll)


midi_dir = "data/pop"
root_dir = os.path.dirname(os.path.abspath(__file__))
piano_roll_dir = os.path.join(root_dir, "data/piano_rolls")

time_steps_per_second = 8
time_steps_per_bar = 16
time_steps_per_beat = 4
time_steps_per_sixteenth_note = 1

os.makedirs(piano_roll_dir, exist_ok=True)

num_files = sum([len(files) for r, d, files in os.walk(midi_dir)])

with tqdm(total=num_files, desc="Processing MIDI files") as pbar:
    for root, dirs, files in os.walk(midi_dir):
        for file in files:
            if not file.endswith('.mid'):
                continue

            midi_path = os.path.join(root, file)
            process_midi_file(midi_path, piano_roll_dir, time_steps_per_second,
                              time_steps_per_bar, time_steps_per_beat)
            pbar.update(1)

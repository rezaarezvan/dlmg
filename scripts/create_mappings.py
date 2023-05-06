import os
import numpy as np
from music21 import converter, chord, note
from tqdm import tqdm

# Define the path to the directory containing the MIDI files
data_dir = '../data/raw_data/'

# Define the sequence length used during training
sequence_length = 100

# Create a list to store all notes and chords from the MIDI files
notes = []

MAX_AMOUNT = 100
counter = 0

# Loop over all MIDI files in the directory
for root, dirs, files in os.walk(data_dir):
    for file in tqdm(files):
        if file.endswith(".mid"):
            # Load the MIDI file
            file_path = os.path.join(root, file)
            try:
                midi = converter.parse(file_path)
            except:
                continue

            # Extract the notes and chords from the MIDI file
            for element in midi.flat:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

            counter += 1
            if counter >= MAX_AMOUNT:
                break

    if counter >= MAX_AMOUNT:
        break

# Create a set of unique notes and chords
unique_notes = sorted(set(notes))

# Create the mappings between notes/chords and integers
note_to_int = dict((note, number) for number, note in enumerate(unique_notes))
int_to_note = dict((number, note) for number, note in enumerate(unique_notes))

# Save the mappings to disk
np.save('../data/note_to_int.npy', note_to_int)
np.save('../data/int_to_note.npy', int_to_note)

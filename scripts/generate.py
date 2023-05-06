import pypianoroll
import pretty_midi
import numpy as np
from keras.models import load_model

model = load_model('../model/model.h5')

# Define generation parameters
SEQUENCE_LENGTH = 1024  # The same as used during training
GENERATE_LENGTH = 500  # The number of notes to generate
TEMPERATURE = 1.0  # Controls the randomness of the generated notes


def generate_notes(seed_notes, model, sequence_length):
    """Generate new notes from the given seed notes using the trained model."""
    # Convert the seed notes into a numpy array
    seed = np.expand_dims(seed_notes, 0)

    # Generate the new notes using the trained model
    generated_notes = []
    for i in range(sequence_length):
        # Predict the probabilities of the next note
        prob = model.predict(seed)[0][-1]
        prob /= np.sum(prob)

        # Sample the next note using the predicted probabilities
        note = np.random.choice(128, p=prob)

        # Add the sampled note to the generated notes list
        generated_notes.append(note)

        # Add the sampled note to the seed and remove the oldest note
        seed = np.concatenate(
            [seed[:, 1:, :], np.expand_dims(np.eye(128)[note], 0)], axis=1)

    return generated_notes


eval_files = open('../data/eval_files.txt', 'r').readlines()
eval_file = np.random.choice(eval_files).strip()
eval_data = np.load(eval_file, allow_pickle=True)

start_index = np.random.randint(0, len(eval_data) - SEQUENCE_LENGTH)
seed_notes = eval_data[start_index:start_index+SEQUENCE_LENGTH].tolist()

# Generate the notes
generated_notes = generate_notes(seed_notes, model, GENERATE_LENGTH)

# Create a piano roll from the generated notes
piano_roll = np.zeros((GENERATE_LENGTH, 128))
for i, note in enumerate(generated_notes[-GENERATE_LENGTH:]):
    if note != 0:
        piano_roll[i, note] = 100

# Create a PrettyMIDI object from the piano roll
midi_data = pypianoroll.Track(piano_roll=piano_roll)
midi_data = pypianoroll.Multitrack(tracks=[midi_data])
midi_data.write('../generated/generated.mid')

print("Generation complete!")

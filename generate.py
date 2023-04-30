import numpy as np
import pypianoroll as ppr
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("model/model.h5")

# Set the length of the seed sequence
seed_length = 16

# Generate a random seed sequence
seed = np.zeros((1, seed_length, 128))
seed[0, :, np.random.randint(0, 128, seed_length)] = 1

# Generate a sequence of notes using the model
num_notes = 5000  # updated
generated_sequence = seed.copy()
for i in range(num_notes):
    prediction = model.predict(generated_sequence)[:, -1, :]
    predicted_note = np.zeros((1, 1, 128))
    predicted_note[:, :, np.argmax(prediction)] = 1
    generated_sequence = np.concatenate(
        (generated_sequence, predicted_note), axis=1)
    generated_sequence = generated_sequence[:, 1:, :]

# Convert the sequence of notes to a MIDI file
track = ppr.BinaryTrack(pianoroll=generated_sequence[0])
multitrack = ppr.Multitrack(tracks=[track], tempo=np.array([120.0]))
multitrack.write("generated.mid")

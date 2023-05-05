import numpy as np
import pypianoroll as ppr
from tensorflow.keras.models import load_model
from midi2audio import FluidSynth


def generate_sequence(model, seed, num_notes):
    generated_sequence = seed.copy()
    for i in range(num_notes):
        prediction = model.predict(generated_sequence)[:, -1, :]
        predicted_note = np.zeros((1, 1, 128))
        predicted_note[:, :, np.argmax(prediction)] = 1
        generated_sequence = np.concatenate(
            (generated_sequence, predicted_note), axis=1)
        generated_sequence = generated_sequence[:, 1:, :]
    return generated_sequence


np.random.seed(42)  # Set the random seed for reproducibility

model = load_model("model/model.h5")

seed_length = 16
num_notes = 5000
output_midi_filename = "generated.mid"
output_wav_filename = "generated.wav"

seed = np.zeros((1, seed_length, 128))
seed[0, :, np.random.randint(0, 128, seed_length)] = 1

generated_sequence = generate_sequence(model, seed, num_notes)

track = ppr.BinaryTrack(pianoroll=generated_sequence[0])
multitrack = ppr.Multitrack(tracks=[track], tempo=np.array([120.0]))
multitrack.write(output_midi_filename)

print(f"Generated MIDI file saved as {output_midi_filename}")

# Convert the MIDI file to a WAV file
fs = FluidSynth()
fs.midi_to_audio(output_midi_filename, output_wav_filename)
print(f"Generated WAV file saved as {output_wav_filename}")

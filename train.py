import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# Set the directory where piano roll data is located
piano_roll_dir = "data/piano_rolls"

# Load a subset of the data
train_files = open("data/train_files.txt", "r").read().splitlines()[:100]
train_data = [np.load(f"{piano_roll_dir}/{file}") for file in train_files]

# Set the time steps for the piano roll
time_steps_per_second = 8
time_steps_per_bar = 16
time_steps_per_beat = 4
time_steps_per_sixteenth_note = 1

# Pad the training data sequences
print("Padding training data...")
max_length = max([len(x) for x in train_data])
padded_train_data = np.zeros((len(train_data), max_length, 128))
for i, seq in enumerate(train_data):

    # Pad the sequences with zeros
    padded_train_data[i, :len(seq)] = 0

print("Padded training data to shape", padded_train_data.shape)

# Define the model architecture
model = Sequential()
model.add(LSTM(512, input_shape=(None, 128), return_sequences=True))
model.add(Dense(128, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Set up a checkpoint to save the model weights during training
checkpoint = ModelCheckpoint(
    "weights.h5", save_weights_only=True, save_best_only=True)

# Train the model
batch_size = 32
num_batches = int(np.ceil(len(padded_train_data) / batch_size))
for i in range(num_batches):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch = padded_train_data[batch_start:batch_end]
    padded_batch = np.zeros((batch_size, max_length, 128))
    for j in range(len(batch)):
        padded_batch[j, :len(batch[j])] = batch[j]
    model.fit(padded_batch, padded_batch, epochs=1, callbacks=[checkpoint])

# Save the final model
model.save("model.h5")

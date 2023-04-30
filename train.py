import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback

# Set the directory where piano roll data is located
piano_roll_dir = "data/piano_rolls"

# Load a subset of the data
num_files = 1000
train_files = open("data/train_files.txt", "r").read().splitlines()[:num_files]
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
    "weights.h5", monitor="loss", save_weights_only=True, save_best_only=True)

# Define a function to print a progress bar


def print_progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=50, fill="█"):
    percent = f"{(iteration / total) * 100:.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()

# Define a callback to print the overall training progress


def print_overall_progress(epoch, logs):
    global batch_size, num_batches, iteration
    iteration = iteration + 1
    print_progress_bar(iteration, num_batches,
                       prefix="Overall Progress:", suffix="Complete")


# Train the model
batch_size = 32
num_batches = int(np.ceil(len(padded_train_data) / batch_size))
iteration = 0
overall_progress_callback = LambdaCallback(on_epoch_end=print_overall_progress)
for i in range(num_batches):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch = padded_train_data[batch_start:batch_end]
    padded_batch = np.zeros((batch_size, max_length, 128))
    for j in range(len(batch)):
        padded_batch[j, :len(batch[j])] = batch[j]
    model.fit(padded_batch, padded_batch, epochs=1, callbacks=[
              checkpoint, overall_progress_callback], verbose=0)

# Save the final model
model.save("model/model.h5")

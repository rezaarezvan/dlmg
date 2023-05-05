import os
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback


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


def load_data(piano_roll_dir, train_files_txt, num_files):
    train_files = open(train_files_txt, "r").read().splitlines()[:num_files]
    train_data = [np.load(os.path.join(piano_roll_dir, file))
                  for file in train_files]
    return train_data


def pad_data(train_data):
    max_length = max([len(x) for x in train_data])
    padded_train_data = np.zeros((len(train_data), max_length, 128))
    for i, seq in enumerate(train_data):
        padded_train_data[i, :len(seq)] = 0
    return padded_train_data


def create_model(input_shape=(None, 128)):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dense(128, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def train_model(model, padded_train_data, batch_size=32, num_epochs=1):
    num_batches = int(np.ceil(len(padded_train_data) / batch_size))

    checkpoint = ModelCheckpoint(
        "weights.h5", monitor="loss", save_weights_only=True, save_best_only=True
    )

    def print_overall_progress(epoch, logs):
        nonlocal iteration
        iteration = iteration + 1
        print_progress_bar(iteration, num_batches,
                           prefix="Overall Progress:", suffix="Complete")

    iteration = 0
    overall_progress_callback = LambdaCallback(
        on_epoch_end=print_overall_progress)
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch = padded_train_data[batch_start:batch_end]
        padded_batch = np.zeros((batch_size, max_length, 128))
        for j in range(len(batch)):
            padded_batch[j, :len(batch[j])] = batch[j]
        model.fit(padded_batch, padded_batch, epochs=num_epochs, callbacks=[
                  checkpoint, overall_progress_callback], verbose=0)

    model.save("model/model.h5")


piano_roll_dir = "data/piano_rolls"
train_files_txt = "data/train_files.txt"
num_files = 1000

train_data = load_data(piano_roll_dir, train_files_txt, num_files)
print("Padding training data...")
padded_train_data = pad_data(train_data)
max_length = padded_train_data.shape[1]
print("Padded training data to shape", padded_train_data.shape)

model = create_model()
train_model(model, padded_train_data)

import os
import numpy as np
from tensorflow.keras.models import load_model


def load_data(piano_roll_dir, val_files_txt, num_files):
    eval_files = open(val_files_txt, "r").read().splitlines()[:num_files]
    eval_data = [np.load(os.path.join(piano_roll_dir, file))
                 for file in eval_files]
    return eval_data


def evaluate_model(model, eval_data):
    total_loss = 0
    for i, seq in enumerate(eval_data):
        padded_seq = np.zeros((1, len(seq), 128))
        padded_seq[0, :len(seq)] = 0
        pred_seq = model.predict(padded_seq)
        loss = binary_crossentropy(pred_seq, padded_seq)
        total_loss += loss
        print(f"Sequence {i+1} Loss: {loss}")
    average_loss = total_loss / len(eval_data)
    return average_loss


def binary_crossentropy(y_pred, y_true):
    return np.sum(np.square(y_pred - y_true))


piano_roll_dir = "data/piano_rolls"
val_files_txt = "data/val_files.txt"
num_files = 10

eval_data = load_data(piano_roll_dir, val_files_txt, num_files)
model = load_model("model/model.h5")

print("Evaluating model...")
average_loss = evaluate_model(model, eval_data)
print(f"Average Loss: {average_loss}")

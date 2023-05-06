import os
from glob import glob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np

# Define paths to data and model files
DATA_DIR = "../data/piano_rolls"
MODEL_FILE = "../model/model.h5"
EVAL_FILE = "../data/eval_files.txt"

SEQUENCE_LENGTH = 1024  # You can adjust this value based on your preference


def load_data(file_list):
    """
    Load the data from the given list of file paths.

    Args:
        file_list (List[str]): List of file paths.

    Returns:
        np.ndarray: Loaded data as a numpy array.
    """
    data = [np.load(file, allow_pickle=True) for file in file_list]
    data = pad_sequences(data, maxlen=SEQUENCE_LENGTH, dtype='float32')
    return data


if __name__ == '__main__':
    # Load the validation data
    with open(EVAL_FILE, "r") as f:
        eval_files = f.read().splitlines()

    eval_files = [os.path.join(file) for file in eval_files]
    eval_data = load_data(eval_files)

    # Load the model and evaluate it
    model = load_model(MODEL_FILE)
    evaluation = model.evaluate(eval_data, eval_data)

    # Print the evaluation metrics
    print("Evaluation metrics:")
    for i, metric_name in enumerate(model.metrics_names):
        print(f"{metric_name}: {evaluation[i]}")

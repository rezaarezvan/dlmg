import numpy as np
from tensorflow.keras.models import load_model

# Set the directory where piano roll data is located
piano_roll_dir = "data/piano_rolls"

# Load a subset of the data for evaluation
eval_files = open("data/val_files.txt", "r").read().splitlines()[:10]
eval_data = [np.load(f"{piano_roll_dir}/{file}") for file in eval_files]

# Load the trained model
model = load_model("model/model.h5")

# Set the time steps for the piano roll
time_steps_per_second = 8
time_steps_per_bar = 16
time_steps_per_beat = 4
time_steps_per_sixteenth_note = 1

# Evaluate the model on the evaluation data
print("Evaluating model...")
total_loss = 0
for i, seq in enumerate(eval_data):

    # Pad the sequence with zeros
    padded_seq = np.zeros((1, len(seq), 128))
    padded_seq[0, :len(seq)] = 0

    # Generate predictions for the sequence
    pred_seq = model.predict(padded_seq)

    # Compute the loss between the predictions and the true sequence
    loss = np.sum(np.square(pred_seq - padded_seq))
    total_loss += loss

    # Convert the predicted and true sequences to binary labels

    print(f"Sequence {i+1} Loss: {loss}")

average_loss = total_loss / len(eval_data)


print(f"Average Loss: {average_loss}")

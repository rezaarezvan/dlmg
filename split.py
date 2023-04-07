import os
import random

data_dir = "data/piano_rolls"

# Get a list of all the files in the data directory
files = os.listdir(data_dir)

# Shuffle the files randomly
random.shuffle(files)

# Split the files into training, validation, and test sets
train_files = files[:int(0.8 * len(files))]
val_files = files[int(0.8 * len(files)):int(0.9 * len(files))]
test_files = files[int(0.9 * len(files)):]

# Save the split lists to disk
with open("data/train_files.txt", "w") as f:
    f.write("\n".join(train_files))
with open("data/val_files.txt", "w") as f:
    f.write("\n".join(val_files))
with open("data/test_files.txt", "w") as f:
    f.write("\n".join(test_files))

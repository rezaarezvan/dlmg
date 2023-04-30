import os
import random
import argparse


def main(data_dir, split=(0.8, 0.1, 0.1)):
    """Splits the given data directory into train, val, and test sets.
    Args:
        data_dir (str): Path to the directory containing the source files.
        split (Tuple): A tuple containing the split proportions for train, val, and test sets.
    """
    assert sum(split) == 1.0, "The sum of split ratios must be equal to 1.0"

    # check if data_dir is valid
    assert os.path.isdir(data_dir), f"{data_dir} is not a valid directory"

    # Get a list of all the files in the data directory
    files = os.listdir(data_dir)

    # Shuffle the files randomly
    random.shuffle(files)

    # Split the files into training, validation, and test sets
    train_files = files[:int(split[0] * len(files))]
    val_files = files[int(split[0] * len(files))
                          :int((split[0] + split[1]) * len(files))]
    test_files = files[int((split[0] + split[1]) * len(files)):]

    # Save the split lists to disk
    with open("data/train_files.txt", "w") as f:
        f.write("\n".join(train_files))
    with open("data/val_files.txt", "w") as f:
        f.write("\n".join(val_files))
    with open("data/test_files.txt", "w") as f:
        f.write("\n".join(test_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/piano_rolls",
                        help="Path to the directory containing the source files.")
    parser.add_argument("--split", type=float, nargs=3, default=[
                        0.8, 0.1, 0.1], help="A tuple containing the split proportions for train, val, and test sets.")

    args = parser.parse_args()

    main(data_dir=args.data_dir, split=args.split)

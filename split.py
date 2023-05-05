import os
import random
import argparse


def split_data(data_dir, split):
    assert sum(split) == 1.0, "The sum of split ratios must be equal to 1.0"
    assert os.path.isdir(data_dir), f"{data_dir} is not a valid directory"

    files = os.listdir(data_dir)
    random.shuffle(files)

    train_files = files[:int(split[0] * len(files))]
    val_files = files[int(split[0] * len(files)):int((split[0] + split[1]) * len(files))]
    test_files = files[int((split[0] + split[1]) * len(files)):]

    return train_files, val_files, test_files


def save_splits(data_dir, train_files, val_files, test_files):
    with open(os.path.join(data_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(data_dir, "val_files.txt"), "w") as f:
        f.write("\n".join(val_files))
    with open(os.path.join(data_dir, "test_files.txt"), "w") as f:
        f.write("\n".join(test_files))


def main(data_dir, split):
    train_files, val_files, test_files = split_data(data_dir, split)
    save_splits(data_dir, train_files, val_files, test_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/piano_rolls",
                        help="Path to the directory containing the source files.")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help="A tuple containing the split proportions for train, val, and test sets.")

    args = parser.parse_args()
    main(data_dir=args.data_dir, split=args.split)

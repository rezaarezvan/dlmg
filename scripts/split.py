import numpy as np
from typing import Tuple, List
from pathlib import Path
from tqdm import tqdm

data_path = Path("../data/piano_rolls")
train_split = 0.8
eval_split = 0.1
test_split = 0.1

assert train_split + eval_split + test_split == 1.0, "Splits should sum up to 1.0"


def split_data(piano_roll_files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split the list of piano roll files into training, evaluation, and testing sets.

    Args:
        piano_roll_files (List[Path]): List of piano roll file paths.

    Returns:
        Tuple[List[Path], List[Path], List[Path]]: Lists of file paths for training, evaluation, and testing sets.
    """
    num_files = len(piano_roll_files)
    indices = np.arange(num_files)
    np.random.shuffle(indices)

    train_end = int(num_files * train_split)
    eval_end = int(num_files * (train_split + eval_split))

    train_indices = indices[:train_end]
    eval_indices = indices[train_end:eval_end]
    test_indices = indices[eval_end:]

    train_files = [piano_roll_files[i] for i in train_indices]
    eval_files = [piano_roll_files[i] for i in eval_indices]
    test_files = [piano_roll_files[i] for i in test_indices]

    return train_files, eval_files, test_files


def main():
    """
    Split the piano roll data into training, evaluation, and testing sets and save the file paths in text files.
    """
    np.random.seed(42)  # Set the random seed for reproducibility

    piano_roll_files = list(data_path.glob("*.npy"))

    print("Splitting data...")
    train_files, eval_files, test_files = split_data(piano_roll_files)

    print("Saving file lists...")
    with open("../data/train_files.txt", "w") as f:
        for file_path in tqdm(train_files, desc="Writing train files"):
            f.write(f"{file_path}\n")

    with open("../data/eval_files.txt", "w") as f:
        for file_path in tqdm(eval_files, desc="Writing eval files"):
            f.write(f"{file_path}\n")

    with open("../data/test_files.txt", "w") as f:
        for file_path in tqdm(test_files, desc="Writing test files"):
            f.write(f"{file_path}\n")

    print("Data split complete.")
    print(f"Training files: {len(train_files)}")
    print(f"Evaluation files: {len(eval_files)}")
    print(f"Testing files: {len(test_files)}")


if __name__ == "__main__":
    main()

import os
import shutil


def move_midi_files(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".mid"):
                src_path = os.path.join(root, filename)
                dest_path = os.path.join(data_dir, filename)
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} to {dest_path}")


if __name__ == "__main__":
    data_dir = "data/pop"
    move_midi_files(data_dir)

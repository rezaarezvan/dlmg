import os
import shutil

data_dir = "data/pop"
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(".mid"):
            src_path = os.path.join(root, filename)
            dest_path = os.path.join(data_dir, filename)
            shutil.move(src_path, dest_path)

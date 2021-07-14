
# requirements
import os

# create a directory
def make_dir(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
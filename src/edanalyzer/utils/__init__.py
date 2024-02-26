import os
from pathlib import Path

def try_make(path):
    try:
        os.mkdir(path)
    except Exception as e:
        return


def try_link(source_path, target_path):
    try:
        os.symlink(Path(source_path).resolve(), target_path)
    except Exception as e:
        # print(e)
        return
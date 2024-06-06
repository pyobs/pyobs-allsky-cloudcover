import os
from copy import copy
from typing import List

import pyobs.images


class ImageIterator:
    def __init__(self, file_paths: List[os.path], start=0, stop=None, step=1):
        self._file_paths = file_paths[start:stop:step]
        self._index = 0

    def __iter__(self):
        return copy(self)

    def __next__(self):
        try:
            file_path = self._file_paths[self._index]
        except IndexError:
            raise StopIteration

        self._index = self._index + 1
        return pyobs.images.Image.from_file(file_path)

    def __len__(self):
        return len(self._file_paths)

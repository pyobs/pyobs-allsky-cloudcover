from __future__ import annotations
from typing import Tuple, Dict, Any


class PreprocessorOptions(object):
    def __init__(self, mask_filepath: str, bin_size: int = 2) -> None:
        self.mask_file_path = mask_filepath
        self.bin_size = bin_size

    @classmethod
    def from_dict(cls, options: Dict[str, Any]) -> PreprocessorOptions:
        return cls(**options)

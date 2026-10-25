from __future__ import annotations

from typing import Any


class PreprocessorOptions:
    """``mask_filepath`` and ``flip_vertical`` only exist for 1.x configs (``preprocessor:`` section)."""

    def __init__(self, mask_filepath: str = "", bin_size: int = 2, flip_vertical: bool = False) -> None:
        self.mask_file_path = mask_filepath
        self.bin_size = bin_size
        self.flip_vertical = flip_vertical

    @classmethod
    def from_dict(cls, options: dict[str, Any]) -> PreprocessorOptions:
        return cls(**options)

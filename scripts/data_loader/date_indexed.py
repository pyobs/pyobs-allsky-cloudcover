import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd


class DateIndexedImageLoader:
    def __init__(self, index_file_path: str):
        self._index_file_path = index_file_path

        self._dates: npt.NDArray[datetime.datetime] | None = None
        self._file_paths: npt.NDArray[str] | None = None

    def load(self) -> None:
        index_df = pd.read_csv(self._index_file_path, index_col=0)

        self._dates = np.array([datetime.datetime.fromisoformat(date) for date in index_df.index])
        self._file_paths = np.array(index_df.path)

    def __call__(
        self, interval: tuple[datetime.datetime, datetime.datetime]
    ) -> tuple[npt.NDArray[datetime.datetime], npt.NDArray[str]]:
        interval_filter = (self._dates >= interval[0]) & (self._dates <= interval[1])
        return self._dates[interval_filter], self._file_paths[interval_filter]

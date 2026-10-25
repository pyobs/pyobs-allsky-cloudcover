import abc

import numpy as np
import numpy.typing as npt


class StarDetector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, image: npt.NDArray[np.float64], median_limit: float | None = None) -> bool:
        """Whether the window contains a star. Windows whose median exceeds ``median_limit`` never do."""

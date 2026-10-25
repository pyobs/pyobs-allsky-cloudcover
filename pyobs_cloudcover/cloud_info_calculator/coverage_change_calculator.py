
import numpy as np
import numpy.typing as npt


class CoverageChangeCalculator:
    def __init__(self) -> None:
        self._last_map: npt.NDArray[np.float64] | None = None

    def __call__(self, new_map: npt.ArrayLike) -> float | None:
        """Fraction of positions known in both maps whose cloud state changed. Maps hold 1.0/0.0/NaN."""
        new = np.array(new_map, dtype=np.float64)

        last = None if self._last_map is None else np.array(self._last_map, dtype=np.float64)
        self._last_map = new
        if last is None or last.shape != new.shape:
            return None

        known = ~np.isnan(last) & ~np.isnan(new)
        if not np.any(known):
            return None

        changed = np.logical_xor(new[known].astype(np.bool_), last[known].astype(np.bool_))
        return float(np.mean(changed))

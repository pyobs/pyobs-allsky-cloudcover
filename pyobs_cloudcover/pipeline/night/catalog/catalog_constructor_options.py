from __future__ import annotations

from typing import Any


class CatalogConstructorOptions:
    def __init__(
        self, filepath: str = "", alt_filter: float = 20.0, v_mag_filter: float = 7.5, distance_filter: float = 0.0
    ) -> None:
        self.filepath = filepath
        self.alt_filter = alt_filter
        self.v_mag_filter = v_mag_filter
        self.distance_filter = distance_filter

    @classmethod
    def from_dict(cls, options: dict[str, Any]) -> CatalogConstructorOptions:
        filepath: str = str(options.get("filepath") or "")

        if "filter" not in options:
            return cls(filepath)

        filter_options: dict[str, Any] = options["filter"]

        filter_kwargs: dict[str, float] = {}

        if "alt" in filter_options:
            filter_kwargs["alt_filter"] = filter_options["alt"]

        if "v_mag" in filter_options:
            filter_kwargs["v_mag_filter"] = filter_options["v_mag"]

        if "distance" in filter_options:
            filter_kwargs["distance_filter"] = filter_options["distance"]

        return cls(filepath, **filter_kwargs)

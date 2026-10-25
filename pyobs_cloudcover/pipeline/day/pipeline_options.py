from __future__ import annotations

from typing import Any

from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_options import CloudMapGeneratorOptions

# blue/red + blue/green below which a pixel is cloudy; camera specific (this suits the simulated presets)
THRESHOLD_DEFAULT = 1.85


class DayPipelineOptions:
    """Options of the day pipeline.

    A 1.x config is recognised by ``mask_filepath``; its ``world_model`` and mask are in raw full-resolution
    pixels and images without a Bayer keyword are debayered as RGGB, as in 1.x. Otherwise the world model and
    mask come from the camera profile and ``bin_size`` sets the binning of the colour-ratio map.
    """

    def __init__(
        self,
        model_options: dict[str, Any],
        mask_filepath: str,
        sun_apparent_size: float,
        cloud_map: CloudMapGeneratorOptions,
        bin_size: int = 1,
        legacy: bool = False,
    ) -> None:
        self.model_options = model_options
        self.cloud_map = cloud_map
        self.mask_filepath = mask_filepath
        self.sun_apparent_size = sun_apparent_size
        self.bin_size = bin_size
        self.legacy = legacy

    @classmethod
    def from_dict(cls, options: dict[str, Any]) -> DayPipelineOptions:
        legacy = "mask_filepath" in options
        return DayPipelineOptions(
            model_options=options.get("world_model") or {},
            mask_filepath=str(options.get("mask_filepath") or ""),
            sun_apparent_size=float(options.get("sun_apparent_size", 0.54)),
            cloud_map=CloudMapGeneratorOptions.from_dict(
                options.get("cloud_map", {"threshold": 3.5 if legacy else THRESHOLD_DEFAULT})
            ),
            bin_size=int(options.get("bin_size", 1 if legacy else 2)),
            legacy=legacy,
        )

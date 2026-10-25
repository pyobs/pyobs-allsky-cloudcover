from __future__ import annotations

from typing import Any

from pyobs_cloudcover.pipeline.night.altaz_grid_generator.altaz_grid_options import AltAzGridOptions
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor_options import CatalogConstructorOptions
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_options import CloudMapGeneratorOptions
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator_options import (
    LimMagnitudeMapGeneratorOptions,
)
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor_options import PreprocessorOptions
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_options import StarReverseMatcherOptions

# defaults chosen on simulated images (benchmarks/night_method.py); thresholds are camera specific
DETECTOR_DEFAULTS: dict[str, Any] = {"type": "aperture", "snr": 3.0}
ESTIMATOR_DEFAULT = "robust"
THRESHOLD_DEFAULT = 4.8


class NightPipelineOptions:
    """Options of the night pipeline.

    Two config styles are understood. In the current one, the world model and sky mask come from the camera
    profile (canonical, full-resolution pixels) and the pipeline has ``bin_size``, ``detector`` and
    ``lim_mag_map.estimator``. A 1.x config is recognised by its ``preprocessor:`` section; it keeps its 1.x
    meaning: the image is flipped vertically and multiplied by ``preprocessor.mask_filepath``, the
    ``world_model`` is in the coordinates of that flipped, binned image, and the 1.x detector (``reverse_matcher``)
    and border estimator stay the defaults, so tuned thresholds keep their meaning.
    """

    def __init__(
        self,
        model_options: dict[str, Any],
        preprocessor_options: PreprocessorOptions,
        catalog_options: CatalogConstructorOptions,
        star_matcher_options: StarReverseMatcherOptions,
        lim_mag_map_generator_options: LimMagnitudeMapGeneratorOptions,
        cloud_generator_options: CloudMapGeneratorOptions,
        altaz_grid_options: AltAzGridOptions,
        moon_apparent_size: float,
        legacy: bool = False,
    ) -> None:

        self.model_options = model_options
        self.altaz_grid_options = altaz_grid_options
        self.preprocessor_options = preprocessor_options
        self.catalog_options = catalog_options
        self.star_matcher_options = star_matcher_options
        self.lim_mag_map_generator_options = lim_mag_map_generator_options
        self.cloud_generator_options = cloud_generator_options
        self.moon_apparent_size = moon_apparent_size
        self.legacy = legacy

    @classmethod
    def from_dict(cls, options: dict[str, Any]) -> NightPipelineOptions:
        legacy = "preprocessor" in options
        model_options = options.get("world_model") or {}
        if legacy:
            preprocessor_options = PreprocessorOptions(**options["preprocessor"], flip_vertical=True)
            matcher = dict(options.get("reverse_matcher", {}))
            lim_mag = {"estimator": "border", **options.get("lim_mag_map", {})}
            cloud_map = options.get("cloud_map", {})
        else:
            preprocessor_options = PreprocessorOptions(bin_size=int(options.get("bin_size", 2)))
            if "reverse_matcher" in options and "detector" not in options:
                matcher = dict(options["reverse_matcher"])
            else:
                matcher = {**DETECTOR_DEFAULTS, **options.get("detector", {})}
            lim_mag = {"estimator": ESTIMATOR_DEFAULT, **options.get("lim_mag_map", {})}
            cloud_map = {"threshold": THRESHOLD_DEFAULT, **options.get("cloud_map", {})}

        return cls(
            model_options,
            preprocessor_options,
            CatalogConstructorOptions.from_dict(options.get("catalog", {})),
            StarReverseMatcherOptions.from_dict(matcher),
            LimMagnitudeMapGeneratorOptions.from_dict(lim_mag),
            CloudMapGeneratorOptions.from_dict(cloud_map),
            AltAzGridOptions.from_dict(options.get("altaz_grid", {})),
            float(options.get("moon_apparent_size", 10.0)),
            legacy,
        )

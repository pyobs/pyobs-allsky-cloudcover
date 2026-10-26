import numpy as np
from astroplan import Observer

from pyobs_cloudcover.camera.profile import CameraProfile
from pyobs_cloudcover.cloud_info_calculator import CloudInfoCalculatorFactory
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.altaz_map_generator_factory import AltAzMapGeneratorFactory
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor_factory import CatalogConstructorFactory
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_factory import CloudMapGeneratorFactory
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator_factory import (
    LimMagnitudeMapGeneratorFactory,
)
from pyobs_cloudcover.pipeline.night.moon_masker import MoonMasker
from pyobs_cloudcover.pipeline.night.photometry import PhotometricCalibrator
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline, PhotometrySettings
from pyobs_cloudcover.pipeline.night.pipeline_options import NightPipelineOptions
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor_factory import PreprocessorFactory
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher_factory import StarReverseMatcherFactory
from pyobs_cloudcover.world_model import ScaledModel, WorldModel
from pyobs_cloudcover.world_model.world_model_factory import WorldModelFactory


class NightPipelineFactory:
    def __init__(self, observer: Observer, camera: CameraProfile | None = None):
        self._observer = observer
        self._camera = camera or CameraProfile()

    def world_model(self, options: NightPipelineOptions) -> WorldModel:
        """The model in the coordinates of the binned image the pipeline analyses."""
        world_model_factory = WorldModelFactory(self._observer)
        if options.legacy:
            # 1.x: the model was fitted on the flipped, binned image
            return world_model_factory(options.model_options)
        config = options.model_options or self._camera.config.world_model
        if not config:
            raise ValueError("No world model: set camera.world_model (or pipelines.night.options.world_model)")
        return ScaledModel(world_model_factory(config), options.preprocessor_options.bin_size)

    def __call__(self, options: NightPipelineOptions) -> NightPipeline:
        model = self.world_model(options)

        preprocessor = PreprocessorFactory(options.preprocessor_options)()
        catalog_constructor = CatalogConstructorFactory(options.catalog_options, model, self._observer)()
        altaz_map_generator = AltAzMapGeneratorFactory(options.altaz_grid_options)()
        star_reverse_matcher = StarReverseMatcherFactory(options.star_matcher_options)()
        lim_mag_map_generator = LimMagnitudeMapGeneratorFactory(options.lim_mag_map_generator_options)()
        cloud_map_generator = CloudMapGeneratorFactory(options.cloud_generator_options)()
        coverage_info_calculator = CloudInfoCalculatorFactory()()
        moon_masker = MoonMasker(options.moon_apparent_size, self._observer)

        photometry = None
        ph = options.photometry
        if ph is not None and ph.get("enabled", True) and options.star_matcher_options.type == "aperture":
            unknown = set(ph) - {"enabled", "extinction_coefficient", "zero_point", "radius", "cloud_threshold",
                                 "cloud_map"}
            if unknown:
                raise ValueError(f"Unknown photometry option(s): {sorted(unknown)}")
            photometry = PhotometrySettings(
                PhotometricCalibrator(float(ph["extinction_coefficient"]), ph["zero_point"]),
                np.deg2rad(float(ph["radius"])),
                float(ph["cloud_threshold"]),
                bool(ph["cloud_map"]),
            )

        return NightPipeline(
            preprocessor,
            catalog_constructor,
            altaz_map_generator,
            star_reverse_matcher,
            lim_mag_map_generator,
            cloud_map_generator,
            moon_masker,
            coverage_info_calculator,
            sky_mask=None if options.legacy else self._camera.mask,
            photometry=photometry,
            model=model,
        )

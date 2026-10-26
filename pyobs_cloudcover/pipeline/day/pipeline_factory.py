import numpy as np
from astroplan import Observer

from pyobs_cloudcover.camera.profile import CameraProfile
from pyobs_cloudcover.cloud_info_calculator import CloudInfoCalculatorFactory
from pyobs_cloudcover.pipeline.day.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.day.pipeline import DayPipeline
from pyobs_cloudcover.pipeline.day.pipeline_options import DayPipelineOptions
from pyobs_cloudcover.pipeline.day.sun_masker import SunMasker
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator_factory import CloudMapGeneratorFactory
from pyobs_cloudcover.world_model import ScaledModel
from pyobs_cloudcover.world_model.world_model_factory import WorldModelFactory


class DayPipelineFactory:
    def __init__(self, observer: Observer, camera: CameraProfile | None = None):
        self._observer = observer
        self._camera = camera or CameraProfile()

    def __call__(self, options: DayPipelineOptions) -> DayPipeline:
        config = options.model_options or (None if options.legacy else self._camera.config.world_model)
        if not config:
            raise ValueError("No world model: set camera.world_model (or pipelines.day.options.world_model)")
        model = ScaledModel(WorldModelFactory(self._observer)(config), options.bin_size)

        if options.legacy:
            mask = np.load(options.mask_filepath) if options.mask_filepath else None
        else:
            mask = self._camera.mask

        return DayPipeline(
            mask,
            AltAzMapGenerator(model, 0),
            CloudMapGeneratorFactory(options.cloud_map)(),
            SunMasker(options.sun_apparent_size, self._observer),
            CloudInfoCalculatorFactory()(),
            bin_size=options.bin_size,
            # 1.x always debayered as RGGB at the array origin, whatever the header said; on the original
            # camera that was right and its header (BAYERPAT=GBRG, no ROWORDER) was not
            force_bayer="RGGB" if options.legacy else None,
        )

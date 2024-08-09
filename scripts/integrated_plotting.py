import sys
import datetime

import numpy as np
from astroplan import Observer
from astropy.visualization import PercentileInterval
import astropy.units as u
from pyobs.images import Image

from pyobs_cloudcover.pipeline.day.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.cloud_info_calculator import CoverageCalculator
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.cloud_info_calculator import ZenithCloudCoverageCalculator
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator import LimMagnitudeMapGenerator
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline

from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.detector.sigma_treshhold_detector import \
    SigmaThresholdDetector
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StarReverseMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow
from pyobs_cloudcover.world_model.wcs_model_loader import WCSModelLoader


def main() -> None:
    import matplotlib.pyplot as plt

    wcs_file, catalog_file = sys.argv[1:3]
    image_files = sys.argv[3:]

    images = [Image.from_file(image_file) for image_file in image_files]
    obs_times = [datetime.datetime.fromisoformat(image.header["DATE-OBS"]) for image in images]

    pipeline, preprocessor = build_pipeline(wcs_file, catalog_file, images[0].data.shape, len(images))

    start = datetime.datetime.now()
    cloud_coverage_infos = [pipeline(image.data, obs_time) for image, obs_time in zip(images, obs_times)]
    end = datetime.datetime.now()

    print((end-start).total_seconds())

    cutoff = 5.5

    plt.figure(figsize=(20, 20))
    plt.subplot(121)
    plt.imshow(PercentileInterval(99)(preprocessor(images[-1].data)), cmap="gray")
    plt.subplot(122)
    plt.title(obs_times[0].isoformat())
    plt.imshow(PercentileInterval(99)(preprocessor(images[-1].data)), cmap="gray")
    plt.imshow(cloud_coverage_infos[-1].cloud_cover_map, alpha=(cloud_coverage_infos[-1].cloud_cover_map < cutoff).astype(np.float_))
    plt.colorbar()

    plt.show()


def build_pipeline(wcs_file, catalog_file, image_shape, integration_size) -> (NightPipeline, Preprocessor):
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)

    wcs_model_factory = WCSModelLoader(wcs_file)
    model = wcs_model_factory()

    mask = ImageMasker(np.ones(image_shape).astype(np.bool_))
    binner = ImageBinner(2)
    preprocessor = Preprocessor(mask, binner)

    altaz_catalog_loader = AltAzCatalogLoader.from_csv(catalog_file)
    catalog_constructor = CatalogConstructor(altaz_catalog_loader, model, observer, 30.0, 7.0, 4.0)

    altaz_list_generator = AltAzMapGenerator(model, 30.0)

    reverse_matcher = StarReverseMatcher(SigmaThresholdDetector(3.0, 4.0, 3e8), ImageWindow(10.0))

    cloud_map_gem = LimMagnitudeMapGenerator(7.0, integration_size)

    coverage_calculator = CoverageCalculator(5.5)
    coverage_change_calculator = CoverageChangeCalculator(5.5)
    zenith_masker = ZenithCloudCoverageCalculator(80)
    cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_calculator, coverage_change_calculator,
                                                            zenith_masker)

    pipeline = NightPipeline(preprocessor, catalog_constructor, altaz_list_generator, reverse_matcher, cloud_map_gem,
                             cloud_coverage_info_calculator)

    return pipeline, preprocessor


if __name__ == '__main__':
    main()

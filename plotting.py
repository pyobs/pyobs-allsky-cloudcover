import sys
import datetime

import numpy as np
from astroplan import Observer
from astropy.visualization import PercentileInterval
import astropy.units as u
from pyobs.images import Image

from pyobs_cloudcover.pipeline.night.altaz_map_generator.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_calculator import CoverageCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.coverage_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.night.cloud_coverage_calculator.zenith_masker import ZenithMasker
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline
from pyobs_cloudcover.pipeline.night.preprocessor.background_remover import BackgroundRemover
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

    wcs_file, catalog_file, good_image_file, bad_image_file = sys.argv[1:]

    good_image = Image.from_file(good_image_file)
    good_obs_time = datetime.datetime.fromisoformat(good_image.header["DATE-OBS"])

    pipeline = build_pipeline(wcs_file, catalog_file, good_image.data.shape)
    good_coverage_info = pipeline(good_image.data, good_obs_time)

    bad_image = Image.from_file(bad_image_file)
    bad_obs_time = datetime.datetime.fromisoformat(bad_image.header["DATE-OBS"])
    bad_coverage_info = pipeline(bad_image.data, bad_obs_time)

    cutoff = 5.8

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(PercentileInterval(99)(ImageBinner(2)(good_image.data)), cmap="gray")
    plt.imshow(good_coverage_info.cloud_cover_map, alpha=(good_coverage_info.cloud_cover_map < cutoff).astype(np.float_))
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(PercentileInterval(99)(ImageBinner(2)(bad_image.data)), cmap="gray")
    plt.imshow(bad_coverage_info.cloud_cover_map, alpha=(bad_coverage_info.cloud_cover_map < cutoff).astype(np.float_))
    plt.colorbar()

    plt.show()

def build_pipeline(wcs_file, catalog_file, image_shape) -> NightPipeline:
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)

    wcs_model_factory = WCSModelLoader(wcs_file)
    model = wcs_model_factory()

    mask = ImageMasker(np.ones(image_shape).astype(np.bool_))
    binner = ImageBinner(2)
    background_remover = BackgroundRemover(3.0, (5, 5))
    preprocessor = Preprocessor(mask, binner, background_remover)

    altaz_catalog_loader = AltAzCatalogLoader.from_csv(catalog_file)
    catalog_constructor = CatalogConstructor(altaz_catalog_loader, model, observer, 30.0, 7.0, 4.0)

    altaz_list_generator = AltAzMapGenerator(model, 30.0)

    reverse_matcher = StarReverseMatcher(SigmaThresholdDetector(3.0), ImageWindow(6.0))

    cloud_map_gem = CloudMapGenerator(7.0)

    coverage_calculator = CoverageCalculator(0.5)
    coverage_change_calculator = CoverageChangeCalculator()
    zenith_masker = ZenithMasker(80)
    cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_calculator, coverage_change_calculator,
                                                            zenith_masker)

    pipeline = NightPipeline(preprocessor, catalog_constructor, altaz_list_generator, reverse_matcher, cloud_map_gem,
                             cloud_coverage_info_calculator)

    return pipeline

if __name__ == '__main__':
    main()
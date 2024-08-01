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

    wcs_file, catalog_file, good_image_file, bad_image_file = sys.argv[1:]

    good_image = Image.from_file(good_image_file)
    good_obs_time = datetime.datetime.fromisoformat(good_image.header["DATE-OBS"])

    pipeline, preprocessor = build_pipeline(wcs_file, catalog_file, good_image.data.shape)

    start = datetime.datetime.now()
    good_coverage_info = pipeline(good_image.data, good_obs_time)
    end = datetime.datetime.now()

    print((end-start).total_seconds())

    bad_image = Image.from_file(bad_image_file)
    bad_obs_time = datetime.datetime.fromisoformat(bad_image.header["DATE-OBS"])
    bad_coverage_info = pipeline(bad_image.data, bad_obs_time)

    print(good_coverage_info.zenith_cover, bad_coverage_info.zenith_cover)

    cutoff = 5.5

    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.title(good_obs_time.isoformat())
    plt.imshow(PercentileInterval(99)(preprocessor(good_image.data)), cmap="gray")
    plt.imshow(good_coverage_info.cloud_cover_map, alpha=(good_coverage_info.cloud_cover_map < cutoff).astype(np.float_))
    plt.colorbar()

    plt.subplot(222)
    plt.title(bad_obs_time.isoformat())
    plt.imshow(PercentileInterval(99)(preprocessor(bad_image.data)), cmap="gray")
    plt.imshow(bad_coverage_info.cloud_cover_map, alpha=(bad_coverage_info.cloud_cover_map < cutoff).astype(np.float_))
    plt.colorbar()

    plt.subplot(223)
    plt.imshow(PercentileInterval(99)(preprocessor(good_image.data)), cmap="gray")


    plt.subplot(224)
    plt.imshow(PercentileInterval(99)(preprocessor(bad_image.data)), cmap="gray")
    plt.imshow(good_coverage_info.cloud_cover_map,
               alpha=np.logical_xor((good_coverage_info.cloud_cover_map < cutoff), (bad_coverage_info.cloud_cover_map < cutoff)).astype(np.float_))

    plt.show()

def build_pipeline(wcs_file, catalog_file, image_shape) -> (NightPipeline, Preprocessor):
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)

    wcs_model_factory = WCSModelLoader(wcs_file)
    model = wcs_model_factory()

    mask = ImageMasker(np.ones(image_shape).astype(np.bool_))
    binner = ImageBinner(2)
    preprocessor = Preprocessor(mask, binner)

    altaz_catalog_loader = AltAzCatalogLoader.from_csv(catalog_file)
    catalog_constructor = CatalogConstructor(altaz_catalog_loader, model, observer, 30.0, 7.0, 4.0)

    altaz_list_generator = AltAzMapGenerator(model, 30.0)

    reverse_matcher = StarReverseMatcher(SigmaThresholdDetector(4.0, 4.0, 7e5), ImageWindow(10.0))

    cloud_map_gem = LimMagnitudeMapGenerator(7.0)

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

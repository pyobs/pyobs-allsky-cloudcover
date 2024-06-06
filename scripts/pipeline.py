import datetime
import sys

import astropy.units as u
import numpy as np
from astroplan import Observer

from data_loader.sun_alt_date_filter import SunAltDateFilter
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
from scripts.data_loader.date_indexed import DateIndexedImageLoader
from scripts.data_loader.image_iterator import ImageIterator


def main() -> None:
    wcs_file, catalog_file, index_file, start, end, output = sys.argv[1:]
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)

    sun_alt_filter = SunAltDateFilter(observer)
    loader = DateIndexedImageLoader(index_file)
    loader.load()
    dates, file_paths = loader(interval=(datetime.datetime(2024, 3, 4), datetime.datetime(2024, 3, 31)))

    filtered_dates = sun_alt_filter(dates)
    filtered_file_paths = file_paths[np.isin(dates, filtered_dates)]
    image_loader = ImageIterator(filtered_file_paths)

    pipeline = build_pipeline(wcs_file, catalog_file, observer, (2080, 3096))

    coverage = []
    zenith_coverage = []
    zenith_average = []
    dates = []
    for image in image_loader:
        obs_time = datetime.datetime.fromisoformat(image.header["DATE-OBS"])
        coverage_info = pipeline(image.data, obs_time)

        zenith_coverage.append(coverage_info.zenith_cover)
        zenith_average.append(coverage_info.zenith_average)
        coverage.append(coverage_info.total_cover)
        dates.append(coverage_info.obs_time)

    write_to_file(output, zenith_coverage, zenith_average, coverage, dates)


def build_pipeline(wcs_file, catalog_file, observer, image_shape) -> NightPipeline:
    wcs_model_factory = WCSModelLoader(wcs_file)
    model = wcs_model_factory()

    mask = ImageMasker(np.ones(image_shape).astype(np.bool_))
    binner = ImageBinner(2)
    background_remover = BackgroundRemover(3.0, (5, 5))
    preprocessor = Preprocessor(mask, binner, background_remover)

    altaz_catalog_loader = AltAzCatalogLoader.from_csv(catalog_file)
    catalog_constructor = CatalogConstructor(altaz_catalog_loader, model, observer, 30.0, 7.0, 4.0)

    altaz_list_generator = AltAzMapGenerator(model, 30.0)

    reverse_matcher = StarReverseMatcher(SigmaThresholdDetector(3.0, 4.0, 7e3), ImageWindow(10.0))

    cloud_map_gem = CloudMapGenerator(7.0)

    coverage_calculator = CoverageCalculator(5.5)
    coverage_change_calculator = CoverageChangeCalculator()
    zenith_masker = ZenithMasker(80)
    cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_calculator, coverage_change_calculator,
                                                            zenith_masker)

    pipeline = NightPipeline(preprocessor, catalog_constructor, altaz_list_generator, reverse_matcher, cloud_map_gem,
                             cloud_coverage_info_calculator)

    return pipeline


def write_to_file(output_file: str, zenith_coverages, zenith_averages, coverages, dates) -> None:
    lines = [
        f"{date.isoformat()},{coverage},{zenith_coverage},{zenith_average}\n"
        for coverage, zenith_coverage, zenith_average, date in zip(coverages, zenith_coverages, zenith_averages, dates)
    ]

    with open(output_file, 'w+') as file:
        file.writelines(lines)


if __name__ == '__main__':
    main()

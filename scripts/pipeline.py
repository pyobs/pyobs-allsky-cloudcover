import datetime
import sys

import astropy.units as u
import numpy as np
from astroplan import Observer

from data_loader.sun_alt_date_filter import SunAltDateFilter
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
from scripts.data_loader.date_indexed import DateIndexedImageLoader
from scripts.data_loader.image_iterator import ImageIterator


def main() -> None:
    wcs_file, catalog_file, index_file, start, end, output = sys.argv[1:]
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)

    sun_alt_filter = SunAltDateFilter(observer)
    loader = DateIndexedImageLoader(index_file)
    loader.load()

    dates, file_paths = loader(interval=(parse_datetime(start), parse_datetime(end)))

    filtered_dates = sun_alt_filter(dates)
    filtered_file_paths = file_paths[np.isin(dates, filtered_dates)]
    image_loader = ImageIterator(filtered_file_paths)

    pipeline = build_pipeline(wcs_file, catalog_file, observer, (2080, 3096))

    coverage = []
    average = []
    std = []
    zenith_coverage = []
    zenith_average = []
    zenith_std = []
    change = []
    dates = []
    for image in image_loader:
        obs_time = datetime.datetime.fromisoformat(image.header["DATE-OBS"])
        print(obs_time)
        coverage_info = pipeline(image.data, obs_time)

        coverage.append(coverage_info.total_cover)
        average.append(coverage_info.average)
        std.append(coverage_info.std)
        zenith_coverage.append(coverage_info.zenith_cover)
        zenith_average.append(coverage_info.zenith_average)
        zenith_std.append(coverage_info.zenith_std)
        change.append(coverage_info.change)
        dates.append(coverage_info.obs_time)

    write_to_file(output, coverage, average, std, zenith_coverage, zenith_average, zenith_std, change, dates)


def build_pipeline(wcs_file, catalog_file, observer, image_shape) -> NightPipeline:
    wcs_model_factory = WCSModelLoader(wcs_file)
    model = wcs_model_factory()

    mask = ImageMasker(np.ones(image_shape).astype(np.bool_))
    binner = ImageBinner(2)
    preprocessor = Preprocessor(mask, binner)

    altaz_catalog_loader = AltAzCatalogLoader.from_csv(catalog_file)
    catalog_constructor = CatalogConstructor(altaz_catalog_loader, model, observer, 30.0, 7.0, 4.0)

    altaz_list_generator = AltAzMapGenerator(model, 30.0)

    reverse_matcher = StarReverseMatcher(SigmaThresholdDetector(3.0, 4.0, 7e3), ImageWindow(10.0))


    cloud_map_gem = LimMagnitudeMapGenerator(7.0)
    coverage_calculator = CoverageCalculator(5.5)
    coverage_change_calculator = CoverageChangeCalculator()
    zenith_masker = ZenithCloudCoverageCalculator(80)
    cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_calculator, coverage_change_calculator,
                                                            zenith_masker)

    pipeline = NightPipeline(preprocessor, catalog_constructor, altaz_list_generator, reverse_matcher, cloud_map_gem,
                             cloud_coverage_info_calculator)

    return pipeline


def parse_datetime(datetime_str: str) -> datetime.datetime:
    date = datetime.datetime.fromisoformat(datetime_str)
    date = date.replace(tzinfo=None)
    return date

def write_to_file(output_file: str, coverages, averages, stds, zenith_coverages, zenith_averages, zenith_stds, changes, dates) -> None:
    lines = [
        f"{date.isoformat()},{coverage},{average},{std},{zenith_coverage},{zenith_average},{zenith_std},{change}\n"
        for coverage, average, std, zenith_coverage, zenith_average, zenith_std, change, date in
        zip(coverages, averages, stds, zenith_coverages, zenith_averages, zenith_stds, changes, dates)
    ]

    with open(output_file, 'w+') as file:
        file.writelines(lines)


if __name__ == '__main__':
    main()

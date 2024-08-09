import datetime
import sys

import astropy.units as u
import numpy as np
from astroplan import Observer

from data_loader.sun_alt_date_filter import SunAltDateFilter
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.cloud_info_calculator import ZenithCloudCoverageCalculator
from pyobs_cloudcover.cloud_info_calculator.coverage_change_calculator import \
    CoverageChangeCalculator
from pyobs_cloudcover.pipeline.day.altaz_map_generator import AltAzMapGenerator
from pyobs_cloudcover.pipeline.day.pipeline import DayPipeline
from pyobs_cloudcover.pipeline.day.sun_masker import SunMasker
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.world_model import SimpleModel
from scripts.data_loader.date_indexed import DateIndexedImageLoader
from scripts.data_loader.image_iterator import ImageIterator


def main() -> None:
    mask_file, index_file, start, end, output = sys.argv[1:]
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)

    sun_alt_filter = SunAltDateFilter(observer, threshold=18, reverse=True)
    loader = DateIndexedImageLoader(index_file)
    loader.load()

    dates, file_paths = loader(interval=(parse_datetime(start), parse_datetime(end)))

    filtered_dates = sun_alt_filter(dates)
    filtered_file_paths = file_paths[np.isin(dates, filtered_dates)]
    image_loader = ImageIterator(filtered_file_paths)

    pipeline = build_pipeline(mask_file, observer)

    total_fraction = []
    zenith_fraction = []
    change = []
    dates = []
    times = []
    for image in image_loader:
        obs_time = datetime.datetime.fromisoformat(image.header["DATE-OBS"])
        print(obs_time)
        start = datetime.datetime.now()
        coverage_info = pipeline(image.data, obs_time)
        end = datetime.datetime.now()
        times.append((end - start).total_seconds())

        total_fraction.append(coverage_info.total_cover)
        zenith_fraction.append(coverage_info.zenith_cover)
        change.append(coverage_info.change)
        dates.append(coverage_info.obs_time)

    write_to_file(output, total_fraction, zenith_fraction, change, dates)

    print(np.average(times[0:]))

def parse_datetime(datetime_str: str) -> datetime.datetime:
    date = datetime.datetime.fromisoformat(datetime_str)
    date = date.replace(tzinfo=None)
    return date


def build_pipeline(mask_file, observer) -> DayPipeline:
    mask = np.load(mask_file)

    model_parameters = [4.81426598e-03, 2.00000000e+00, 1.06352627e+03, 7.57115607e+02, 5.11194838e+02]
    model = SimpleModel(*model_parameters)

    alt_az_generator = AltAzMapGenerator(model, 0)
    cloud_map_gen = CloudMapGenerator(3.8)
    sun_masker = SunMasker(observer)

    coverage_change_calculator = CoverageChangeCalculator()
    zenith_masker = ZenithCloudCoverageCalculator(80)
    cloud_coverage_info_calculator = CoverageInfoCalculator(coverage_change_calculator, zenith_masker)

    pipeline = DayPipeline(mask, alt_az_generator, cloud_map_gen, sun_masker, cloud_coverage_info_calculator)

    return pipeline

def write_to_file(output_file: str, coverages, zenith_coverages, changes, dates) -> None:
    lines = [
        f"{date.isoformat()},{coverage},{zenith_coverage},{change}\n"
        for coverage, zenith_coverage, change, date in
        zip(coverages, zenith_coverages, changes, dates)
    ]

    with open(output_file, 'w+') as file:
        file.writelines(lines)


if __name__ == '__main__':
    main()

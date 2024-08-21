import datetime

import numpy as np
from astroplan import Observer

import astropy.units as u

from pyobs_cloudcover.pipeline.day.pipeline_factory import DayPipelineFactory
from pyobs_cloudcover.pipeline.day.pipeline_options import DayPipelineOptions
from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline
from pyobs_cloudcover.pipeline.night.pipeline_factory import NightPipelineFactory
from pyobs_cloudcover.pipeline.night.pipeline_options import NightPipelineOptions
from pyobs_cloudcover.world_model import SimpleModel


def test_day_pipeline() -> None:
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)
    obs_time = datetime.datetime(2024, 3, 9, 1, 48, 48, 297970)

    image = np.zeros((104, 154)).astype(np.uint16) * 1

    image[0, 0] = 0

    pipeline_kwargs = {
        "world_model": {
            "class": "pyobs_cloudcover.world_model.SimpleModel",
            "a0": 4.81426598e-03,
            "F": 2.00000000e+00,
            "R": 1.06352627e+03,
            "c_x": 7.57115607e+01,
            "c_y": 5.11194838e+01
        },
        "mask_filepath": "tests/integration/small_dummy_mask.npy",
        "sun_apparent_size": 0.54,
        "cloud_map": {
            "threshold": 3.5
        }
    }

    pipeline_options = DayPipelineOptions.from_dict(pipeline_kwargs)
    pipeline_factory = DayPipelineFactory(observer)
    pipeline = pipeline_factory(pipeline_options)
    pipeline(image, obs_time)

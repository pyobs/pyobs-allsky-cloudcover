import datetime

import numpy as np
from astroplan import Observer

import astropy.units as u

from pyobs_cloudcover.pipeline.night.pipeline import NightPipeline
from pyobs_cloudcover.pipeline.night.pipeline_factory import NightPipelineFactory
from pyobs_cloudcover.pipeline.night.pipeline_options import NightPipelineOptions
from pyobs_cloudcover.world_model import SimpleModel


def test_night_pipeline() -> None:
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)
    obs_time = datetime.datetime(2024, 3, 9, 1, 48, 48, 297970)

    model_parameters = [4.81426598e-03, 2.00000000e+00, 1.06352627e+03, 7.57115607e+02, 5.11194838e+02]
    model = SimpleModel(*model_parameters)

    stars = np.loadtxt('tests/integration/matches_small_20240308.csv', delimiter=",")

    image = np.zeros((2*1040, 2*1548))

    for star in stars:
        image[int(star[2]), int(star[1])] = 10

    pipeline_kwargs = {
        "preprocessor": {
            "mask_filepath": "tests/integration/mask.npy",
            "bin_size": 2,
            "bkg_sigma_clip": 3.0,
            "bkg_box_size": (5, 5),
        },
        "catalog": {
            "filepath": "tests/integration/catalog.csv",
            "filter": {
                "alt": 30.0,
                "v_mag": 7.0,
                "distance": 0.0
            }
        },
        "reverse_matcher": {
            "sigma_threshold": 3.0,
            "window_size": 6.0
        },
        "cloud_map": {
            "radius": 50.0
        },
        "coverage_info": {
            "cloud_threshold": 0.5,
            "zenith_radius": 20
        }
    }

    pipeline_options = NightPipelineOptions.from_dict(pipeline_kwargs)
    pipeline_factory = NightPipelineFactory(observer, model, pipeline_options)
    pipeline = pipeline_factory()
    pipeline(image, obs_time)

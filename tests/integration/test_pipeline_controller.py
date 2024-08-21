import datetime

import astropy.units as u
import numpy as np
from astroplan import Observer

from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory


def test_night_pipeline() -> None:
    observer = Observer(latitude=51.559299 * u.deg, longitude=9.945472 * u.deg, elevation=201 * u.m)
    obs_time = datetime.datetime(2024, 3, 9, 12, 48, 48, 297970)

    stars = np.loadtxt('tests/integration/matches_small_20240308.csv', delimiter=",")

    image = np.zeros((2*1040, 2*1548))

    for star in stars:
        image[int(star[2]), int(star[1])] = 10

    kwargs = {
            "night": {
                "alt_interval":
                    {
                        "start": None,
                        "end": -18
                    },
                "options":
                    {
                        "world_model": {
                            "class": "pyobs_cloudcover.world_model.SimpleModel",
                            "a0": 4.81426598e-03,
                            "F": 2.00000000e+00,
                            "R": 1.06352627e+03,
                            "c_x": 7.57115607e+02,
                            "c_y": 5.11194838e+02
                        },
                        "preprocessor": {
                            "mask_filepath": "tests/integration/mask.npy",
                            "bin_size": 2
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
                            "distance": 3.0,
                            "median_limit": 9e3,
                            "window_size": 6.0
                        },
                        "altaz_grid": {
                            "point_number": 10,
                            "limiting_altitude": 30
                        },
                        "lim_mag_map": {
                            "radius": 50.0
                        },
                        "cloud_map": {
                            "threshold": 0.5
                        },
                        "moon_apparent_size": 10.0

                    }
            }

         }

    factory = PipelineControllerFactory(observer)
    controller = factory(kwargs)
    assert controller(image, obs_time) is None

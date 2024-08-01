from typing import List, Optional

import numpy as np
from cloudmap_rs import AltAzCoord, SkyPixelQuery

from pyobs_cloudcover.cloud_info_calculator import \
    ZenithCloudCoverageCalculator


def test_zenith_mask() -> None:
    calculator = ZenithCloudCoverageCalculator(altitude=80)

    alt_az_list: List[Optional[AltAzCoord]] = [
        AltAzCoord(np.deg2rad(70), 0), AltAzCoord(np.deg2rad(85), 0), AltAzCoord(np.deg2rad(70), 0),
        AltAzCoord(np.deg2rad(85), 0), AltAzCoord(np.deg2rad(85), 0), AltAzCoord(np.deg2rad(85), 0),
        AltAzCoord(np.deg2rad(70), 0), AltAzCoord(np.deg2rad(85), 0), AltAzCoord(np.deg2rad(70), 0),
    ]

    pixels = [False, True, False, True, True, True, False, True, False]

    sky_query = SkyPixelQuery(alt_az_list, pixels)
    assert calculator(sky_query) == 1

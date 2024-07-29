from pyobs_cloudcover.pipeline.night.altaz_grid_generator.altaz_grid_options import AltAzGridOptions
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator


class AltAzMapGeneratorFactory(object):
    def __init__(self, options: AltAzGridOptions):
        self._point_number = options.point_number
        self._limiting_altitude = options.limiting_altitude

    def __call__(self) -> SphericalAltAzGenerator:
        return SphericalAltAzGenerator(self._point_number, self._limiting_altitude)

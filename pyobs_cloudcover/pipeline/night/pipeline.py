from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.cloud_info_calculator import CoverageInfoCalculator
from pyobs_cloudcover.pipeline.night.aligner import PointingAligner
from pyobs_cloudcover.pipeline.night.altaz_grid_generator.spherical_alt_az_generator import SphericalAltAzGenerator
from pyobs_cloudcover.pipeline.night.catalog.catalog_constructor import CatalogConstructor
from pyobs_cloudcover.pipeline.night.cloud_map_generator.cloud_map_generator import CloudMapGenerator
from pyobs_cloudcover.pipeline.night.lim_magnitude_map_generator.lim_magnitude_map_generator import (
    LimMagnitudeMapGenerator,
)
from pyobs_cloudcover.pipeline.night.moon_masker import MoonMasker
from pyobs_cloudcover.pipeline.night.photometry import PhotometricCalibrator, extinction_map, sky_brightness
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.aperture_matcher import ApertureMatcher
from pyobs_cloudcover.pipeline.night.star_reverse_matcher.star_reverse_matcher import StarReverseMatcher
from pyobs_cloudcover.pipeline.pipeline import Pipeline
from pyobs_cloudcover.sky_query import SkyGrid, SkyPixelQuery
from pyobs_cloudcover.world_model.world_model import WorldModel


@dataclass
class PhotometrySettings:
    calibrator: PhotometricCalibrator
    radius: float  # rad
    cloud_threshold: float = 0.5  # mag of extinction from which a position is cloudy
    cloud_map_from_extinction: bool = False


class NightPipeline(Pipeline):
    def __init__(
        self,
        preprocess: Preprocessor,
        catalog_constructor: CatalogConstructor,
        alt_az_list_generator: SphericalAltAzGenerator,
        star_reverse_matcher: StarReverseMatcher | ApertureMatcher,
        lim_magnitude_map_generator: LimMagnitudeMapGenerator,
        cloud_map_generator: CloudMapGenerator,
        moon_masker: MoonMasker,
        coverage_info_calculator: CoverageInfoCalculator,
        sky_mask: npt.NDArray[np.bool_] | None = None,
        photometry: PhotometrySettings | None = None,
        model: WorldModel | None = None,
        aligner: PointingAligner | None = None,
    ) -> None:
        """``sky_mask``: full-resolution canonical mask (True = sky); stars outside it are ignored.
        ``photometry``: also measure cloud extinction per star (needs the aperture detector); ``model`` (in binned
        pixels) is needed for the sky brightness. ``aligner``: corrects the (shared) world model for small camera
        moves before each frame is analysed."""

        self._preprocess = preprocess
        self._catalog_constructor = catalog_constructor
        self._star_reverse_matcher = star_reverse_matcher
        self._alt_az_list_generator = alt_az_list_generator
        self._lim_magnitude_map_generator = lim_magnitude_map_generator
        self._cloud_map_generator = cloud_map_generator
        self._moon_masker = moon_masker
        self._coverage_info_calculator = coverage_info_calculator
        self._sky_mask = sky_mask
        self._photometry = photometry
        self._model = model
        self._aligner = aligner
        self._binned_mask: npt.NDArray[np.bool_] | None = None

    def _mask(self) -> npt.NDArray[np.bool_] | None:
        if self._sky_mask is not None and self._binned_mask is None:
            # a binned pixel is sky if most of it is
            self._binned_mask = ImageBinner(self._preprocess.bin_size)(self._sky_mask.astype(np.float64)) >= 0.5
        return self._binned_mask

    def _in_image(self, grid: SkyGrid, height: int, width: int) -> npt.NDArray[np.bool_]:
        """Grid positions that land inside the (binned) image."""
        assert self._model is not None
        with np.errstate(invalid="ignore"):
            x, y = (np.asarray(v, dtype=np.float64) for v in self._model.altaz_to_pix(grid.alt, grid.az))
            inside: npt.NDArray[np.bool_] = np.isfinite(x) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
        return inside

    def __call__(self, frame: Frame) -> CloudCoverageInfo:
        obs_time = frame.obs_time
        preprocessed_image = self._preprocess(frame)
        img_height, img_width = preprocessed_image.shape

        mask = self._mask()
        if self._aligner is not None:
            self._aligner(preprocessed_image, obs_time, mask)
        catalog = self._catalog_constructor(obs_time, img_height, img_width)
        if mask is not None:
            catalog.filter_mask(mask)
        matches = self._star_reverse_matcher(preprocessed_image, catalog, frame.saturation)

        grid = self._alt_az_list_generator()

        lim_mag_map = self._lim_magnitude_map_generator(catalog, matches, grid)
        if self._model is not None:
            # positions the camera doesn't see are unknown, not cloudy (cameras that don't cover the whole sky)
            lim_mag_map = np.where(self._in_image(grid, img_height, img_width), lim_mag_map, np.nan)
        cloud_map = self._cloud_map_generator(lim_mag_map)

        extinction_query, zero_point, brightness = None, None, None
        p = self._photometry
        if (
            p is not None
            and isinstance(self._star_reverse_matcher, ApertureMatcher)
            and self._star_reverse_matcher.last
        ):
            measured = self._star_reverse_matcher.last
            exposure = frame.exposure or 1.0
            alt = np.deg2rad(np.asarray(catalog.alt, dtype=np.float64))
            # only frames the limiting-magnitude method calls mostly clear may teach the zero point
            may_learn = bool(np.nanmean(cloud_map) < 0.5) if np.any(np.isfinite(cloud_map)) else False
            stars = p.calibrator(
                measured.signal, measured.noise, np.asarray(catalog.v_mag, float), alt, exposure, may_learn
            )
            ext = extinction_map(grid, alt, np.deg2rad(np.asarray(catalog.az, float)), stars, p.radius)
            extinction_query = self._moon_masker(SkyPixelQuery(grid, ext), obs_time)
            zero_point = stars.reference_zero_point
            if self._model is not None:
                brightness = sky_brightness(preprocessed_image, self._model, zero_point, exposure, mask=mask)
            # without a zero point yet (e.g. the first frames are overcast) the limiting-magnitude map stays
            if p.cloud_map_from_extinction and np.isfinite(zero_point):
                # where no star measures the extinction (e.g. bright moonlight, short exposures), the limiting
                # magnitude decides
                cloud_map = np.where(np.isnan(ext), cloud_map, (ext >= p.cloud_threshold).astype(np.float64))

        sky_query = self._moon_masker(SkyPixelQuery(grid, cloud_map), obs_time)
        lim_mag_query = self._moon_masker(SkyPixelQuery(grid, lim_mag_map), obs_time)

        info = self._coverage_info_calculator(sky_query, obs_time, lim_mag_query, "limiting_magnitude", "night")
        info.extinction_query, info.zero_point, info.sky_brightness = extinction_query, zero_point, brightness
        return info

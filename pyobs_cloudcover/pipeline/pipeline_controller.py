from astroplan import Observer

from pyobs_cloudcover.camera.profile import Frame
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.pipeline.intervall import Interval
from pyobs_cloudcover.pipeline.pipeline import Pipeline


class PipelineController:
    def __init__(self, pipelines: list[Pipeline], sun_alt_intervals: list[Interval], observer: Observer) -> None:
        self._pipelines = pipelines
        self._sun_alt_intervals = sun_alt_intervals
        self._observer = observer
        self.last_reason = ""

        self._check_arg_length()
        self._check_interval_overlap()

    def _check_arg_length(self) -> None:
        if len(self._pipelines) != len(self._sun_alt_intervals):
            raise ValueError("Number of pipelines must equal the intervals")

    def _check_interval_overlap(self) -> None:
        overlap = [
            other.does_intersect(interval)
            for interval in self._sun_alt_intervals
            for other in self._sun_alt_intervals
            if other is not interval
        ]

        if True in overlap:
            raise ValueError("Sun altitude intervals can't overlap!")

    def __call__(self, frame: Frame) -> CloudCoverageInfo | None:
        """Result of the pipeline for the sun's altitude; if there is none, ``last_reason`` says why."""
        sun_alt = self._observer.sun_altaz(frame.obs_time).alt.deg
        self.last_reason = ""

        for pipeline, alt_interval in zip(self._pipelines, self._sun_alt_intervals):
            if sun_alt in alt_interval:
                result = pipeline(frame)
                if result is None:
                    self.last_reason = f"the {type(pipeline).__name__} couldn't analyse this image"
                return result

        self.last_reason = f"no pipeline for the sun at {sun_alt:.1f} deg"
        return None

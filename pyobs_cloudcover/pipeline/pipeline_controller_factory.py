from typing import Dict, Any, List

from astroplan import Observer

from pyobs_cloudcover.pipeline.day.pipeline_factory import DayPipelineFactory
from pyobs_cloudcover.pipeline.day.pipeline_options import DayPipelineOptions
from pyobs_cloudcover.pipeline.intervall import Interval
from pyobs_cloudcover.pipeline.night.pipeline_factory import NightPipelineFactory
from pyobs_cloudcover.pipeline.night.pipeline_options import NightPipelineOptions
from pyobs_cloudcover.pipeline.pipeline import Pipeline
from pyobs_cloudcover.pipeline.pipeline_controller import PipelineController
from pyobs_cloudcover.world_model import WorldModel


class PipelineControllerFactory(object):
    def __init__(self, observer: Observer):
        self._observer = observer
        self._night_pipeline_factory = NightPipelineFactory(observer)
        self._day_pipeline_factory = DayPipelineFactory(observer)

    def __call__(self, pipline_configs: Dict[str, Dict[str, Any]]) -> PipelineController:
        pipelines: List[Pipeline] = []
        intervals: List[Interval] = []

        for pipeline_type, pipeline_config in pipline_configs.items():
            if pipeline_type == 'day':
                day_pipeline_options = DayPipelineOptions.from_dict(pipeline_config["options"])
                pipelines.append(self._day_pipeline_factory(day_pipeline_options))
            elif pipeline_type == 'night':
                night_pipeline_options = NightPipelineOptions.from_dict(pipeline_config["options"])
                pipelines.append(self._night_pipeline_factory(night_pipeline_options))
            else:
                raise ValueError(f"Pipeline type {pipeline_type} not implemented!")

            intervals.append(Interval(**pipeline_config["alt_interval"]))

        return PipelineController(pipelines, intervals, self._observer)

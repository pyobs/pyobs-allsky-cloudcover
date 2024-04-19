from typing import Dict, Any, List

from astroplan import Observer

from pyobs_cloudcover.pipeline.intervall import Interval
from pyobs_cloudcover.pipeline.night.pipeline_factory import NightPipelineFactory
from pyobs_cloudcover.pipeline.night.pipeline_options import NightPipelineOptions
from pyobs_cloudcover.pipeline.pipeline import Pipeline
from pyobs_cloudcover.pipeline.pipeline_controller import PipelineController
from pyobs_cloudcover.world_model import WorldModel


class PipelineControllerFactory(object):
    def __init__(self, observer: Observer, model: WorldModel):
        self._observer = observer
        self._night_pipeline_factory = NightPipelineFactory(observer, model)

    def __call__(self, pipline_configs: Dict[str, Dict[str, Any]]) -> PipelineController:
        pipelines: List[Pipeline] = []
        intervals: List[Interval] = []

        for pipeline_type, pipeline_config in pipline_configs.items():
            if pipeline_type == 'night':
                pipeline_options = NightPipelineOptions.from_dict(pipeline_config["options"])
                pipelines.append(self._night_pipeline_factory(pipeline_options))
            else:
                raise ValueError(f"Pipeline type {pipeline_type} not implemented!")

            intervals.append(Interval(**pipeline_config["alt_interval"]))

        return PipelineController(pipelines, intervals, self._observer)

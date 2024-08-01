from typing import Dict, Any

from astroplan import Observer

from pyobs_cloudcover.pipeline.night.star_reverse_matcher.window import ImageWindow
from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor
from pyobs_cloudcover.web_api.server import Server
from pyobs_cloudcover.world_model import WorldModel


class ServerFactory(object):
    def __init__(self, observer: Observer) -> None:
        self._executor = CoverageQueryExecutor(observer)

    def __call__(self, config: Dict[str, Any]) -> Server:
        server = Server(query_executor=self._executor, **config)
        return server

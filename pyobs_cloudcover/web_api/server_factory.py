from typing import Any

from astroplan import Observer

from pyobs_cloudcover.web_api.coverage_query_executor import CoverageQueryExecutor
from pyobs_cloudcover.web_api.server import Server


class ServerFactory:
    def __init__(self, observer: Observer) -> None:
        self._executor = CoverageQueryExecutor(observer)

    def __call__(self, config: dict[str, Any]) -> Server:
        server = Server(query_executor=self._executor, **config)
        return server

import datetime

from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import SYNCHRONOUS

from pyobs_cloudcover.measurement_log.logger_strategies.measurement_log import LoggerStrategy


class Influx(LoggerStrategy):
    def __init__(self, url: str, bucket: str, org: str, token: str, timeout: float = 10.0) -> None:
        """``timeout`` in seconds for each write."""
        self._client = InfluxDBClient(url=url, token=token, org=org, timeout=int(timeout * 1000))
        self._bucket = bucket
        self._org = org

    def __call__(self, measurements: dict[str, dict[str, float | None]], obs_time: datetime.datetime) -> None:
        points = [self._to_point(name, measurements, obs_time) for name, measurements in measurements.items()]

        with self._client.write_api(write_options=SYNCHRONOUS) as write_api:
            write_api.write(bucket=self._bucket, org=self._org, record=points)

    @staticmethod
    def _to_point(
        name: str, measurements: dict[str, float | None], obs_time: datetime.datetime
    ) -> Point:
        data = Point(name)
        data.time(obs_time)

        for field_name, field_value in measurements.items():
            data.field(field_name, field_value)

        return data

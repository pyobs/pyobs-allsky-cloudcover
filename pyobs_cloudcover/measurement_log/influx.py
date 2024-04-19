import datetime

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.measurement_log import MeasurementLog


class Influx(MeasurementLog):
    def __init__(self, url: str, bucket: str, org: str, token: str) -> None:
        self._client = influxdb_client.InfluxDBClient(
            url=url,
            token=token,
            org=org
        )
        self._bucket = bucket
        self._org = org

    def __call__(self, measurement: CloudCoverageInfo) -> None:
        data = influxdb_client.Point("measurement")

        data.time(measurement.obs_time)

        data.field("total-cover", measurement.total_cover)
        data.field("zenith-cover", measurement.zenith_cover)
        data.field("change", measurement.change)

        with self._client.write_api(write_options=SYNCHRONOUS) as write_api:
            write_api.write(bucket=self._bucket, org=self._org, record=data)

import datetime
from unittest.mock import Mock

import influxdb_client
import numpy as np

from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.measurement_log.influx import Influx

from influxdb_client.client.write_api import WriteApi


def test_call():
    influxdb_client.client.write_api.WriteApi.write = Mock()

    obs_time = datetime.datetime(2020, 1, 1, 0, 0, 0)
    measurement = CloudCoverageInfo(np.array([]), 0, 1, 0.1, obs_time)

    measurement_log = Influx("", "bucket", "org", "token")

    measurement_log(measurement)

    call_args = influxdb_client.client.write_api.WriteApi.write.call_args_list

    kwargs = call_args[0].kwargs
    assert kwargs["bucket"] == "bucket"
    assert kwargs["org"] == "org"

    data: influxdb_client.Point = kwargs["record"]
    assert data._time == obs_time
    assert data._fields["total-cover"] == 0
    assert data._fields["zenith-cover"] == 1
    assert data._fields["change"] == 0.1


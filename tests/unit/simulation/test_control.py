import datetime
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset
from pyobs_cloudcover.simulation.cli import apply_conditions
from pyobs_cloudcover.simulation.control import Conditions, Control

# Sutherland, two days after full moon, moon ~60 deg up
MOONLIT = datetime.datetime(2026, 3, 5, 23, 0, tzinfo=datetime.UTC)


def test_conditions_update() -> None:
    c = Conditions().updated({"cover": "0.7", "moon": "off", "time_offset": -3})
    assert c.cover == 0.7 and c.moon is False and c.time_offset == -3.0
    assert c.updated({"moon": True}).moon is True
    with pytest.raises(ValueError):
        c.updated({"cover": 1.5})
    with pytest.raises(ValueError):
        c.updated({"rain": 1})


def test_moon_off() -> None:
    sim = Simulator(get_preset("zwo-asi224mc", 0.1), seed=1)
    lit = sim.render(MOONLIT)
    assert lit.truth.metadata["moon_alt"] > 30
    sim.moon = False
    dark = sim.render(MOONLIT)
    assert dark.truth.metadata["moon_alt"] < 0
    # no moonlight: a much darker sky and a longer exposure
    assert dark.truth.metadata["sky_zenith_mag"] > lit.truth.metadata["sky_zenith_mag"] + 1
    assert dark.truth.metadata["exposure"] >= lit.truth.metadata["exposure"]


def test_apply_conditions_keeps_the_cloud_pattern() -> None:
    sim = Simulator(get_preset("zwo-asi224mc", 0.1), clouds=[CloudLayer(cover=0.3, seed=4)])
    apply_conditions(sim, Conditions(cover=0.6, optical_depth=1.0, sky_mag=20.0, moon=False), seed=4)
    assert sim.clouds[0].cover == 0.6 and sim.clouds[0].seed == 4 and sim.sky.zenith_mag == 20.0
    assert sim.moon is False
    apply_conditions(sim, Conditions(cover=0.0), seed=4)
    assert sim.clouds == []


def _request(url: str, data: bytes | None = None, content_type: str = "application/json") -> tuple[int, dict]:  # type: ignore[type-arg]
    req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_control_server() -> None:
    control = Control(Conditions())
    server = control.serve("127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(url + "/", timeout=5) as r:
            assert "Cloud cover" in r.read().decode()
        status, state = _request(url + "/conditions")
        assert status == 200 and state["conditions"]["moon"] is True
        status, state = _request(url + "/conditions", json.dumps({"cover": 0.8, "moon": False}).encode())
        assert status == 200 and state["conditions"]["cover"] == 0.8 and control.conditions.moon is False
        assert control.changed.is_set()
        status, _ = _request(url + "/conditions", b"sky_mag=19.5", "application/x-www-form-urlencoded")
        assert status == 200 and np.isclose(control.conditions.sky_mag, 19.5)
        status, error = _request(url + "/conditions", json.dumps({"cover": 2}).encode())
        assert status == 400 and "cover" in error["error"]
    finally:
        server.shutdown()

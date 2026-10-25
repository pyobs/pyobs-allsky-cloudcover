"""End to end: simulated images -> the real module (pyobs, local comm) -> measurement log and web API."""

import asyncio
import datetime
import shutil
from pathlib import Path
from typing import Any

import aiohttp
import pytest
import yaml
from pyobs.events import NewImageEvent
from pyobs.modules import Module

from pyobs_cloudcover.application import Application
from pyobs_cloudcover.measurement_log.logger_strategies.memory import MemoryLog
from pyobs_cloudcover.simulation import CloudLayer, GroundTruth, Simulator, get_preset

ROOT = Path(__file__).parents[2]
NIGHT = datetime.datetime(2026, 3, 20, 22, 0, tzinfo=datetime.UTC)
SCALE = 0.5


def module_config(preset: str, watch_dir: Path, port: int, name: str) -> dict[str, Any]:
    """The preset's example config, pointed at a temp directory, an in-memory measurement log and a free port."""
    config = yaml.safe_load((ROOT / "examples" / f"{preset}.yaml").read_text())
    config.pop("class")
    config["watch"] = {
        "path": str(watch_dir),
        "pattern": config["watch"]["pattern"],
        "poll": True,
        "poll_interval": 0.1,
        "settle_time": 0.2,
    }
    config["camera"]["world_model"] = Simulator(get_preset(preset, SCALE)).world_model.to_config()
    config["measurement_log"]["logger"] = {"type": "memory"}
    config["server"] = {"url": "127.0.0.1", "port": port}
    config["comm"] = {"class": "pyobs.comm.local.LocalComm", "name": name}
    return config


async def wait_for_points(log: MemoryLog, count: int, timeout: float = 30) -> None:
    for _ in range(int(timeout / 0.1)):
        if len(log.points) >= count:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(f"{len(log.points)} of {count} measurements after {timeout} s")


async def get(port: int, path: str) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{port}{path}") as response:
            assert response.status == 200
            return await response.json()  # type: ignore[no-any-return]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["zwo-asi224mc", "zwo-asi178mm"])
async def test_watched_directory_to_influx_and_api(preset: str, tmp_path: Path, unused_tcp_port: int) -> None:
    watch = tmp_path / "images"
    watch.mkdir()
    app = Application(**module_config(preset, watch, unused_tcp_port, f"cloudcover-{preset}"))
    log = app._measurement_log.strategy
    assert isinstance(log, MemoryLog)
    await app.open()
    try:
        # first an overcast frame, then a clear one 10 minutes later; the simulator acts as the camera
        for minutes, cover in ((0, 1.0), (10, 0.0)):
            sim = Simulator(get_preset(preset, SCALE), clouds=[CloudLayer(cover=cover, seed=5)], seed=5)
            frame = sim.render(NIGHT + datetime.timedelta(minutes=minutes))
            frame.write(watch)
            await wait_for_points(log, 1 if cover == 1.0 else 2)

        (t1, overcast), (t2, clear) = log.points
        assert t1 == NIGHT and t2 == NIGHT + datetime.timedelta(minutes=10)
        assert overcast["cloud_total"]["cloudcover"] > 0.7
        assert clear["cloud_total"]["cloudcover"] < 0.3
        assert clear["coverage"]["coverage-change"] > 0.4  # most of the sky cleared up

        # the API answers for the latest (clear) frame
        area = await get(unused_tcp_port, "/query/area?alt=90&az=0&radius=60")
        assert area["obs_time"] == t2.timestamp()
        assert area["fraction"] < 0.3 and area["value"] == pytest.approx(100 * area["fraction"])
        point = await get(unused_tcp_port, "/query/point?alt=60&az=200")
        assert point["value"] in (True, False)
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_new_image_event_input(tmp_path: Path, unused_tcp_port: int) -> None:
    """Images sent by a pyobs camera module (NewImageEvent + VFS) instead of a watched directory."""
    images = tmp_path / "vfs"
    images.mkdir()
    vfs = {
        "class": "pyobs.vfs.VirtualFileSystem",
        "roots": {"cam": {"class": "pyobs.vfs.LocalFile", "root": str(images)}},
    }

    config = module_config("zwo-asi224mc", tmp_path / "unused", unused_tcp_port, "cloudcover-events")
    del config["watch"]
    config["image_sender"] = "camera-e2e"
    app = Application(**config, vfs=vfs)
    camera = Module(comm={"class": "pyobs.comm.local.LocalComm", "name": "camera-e2e"}, vfs=vfs)
    await app.open()
    await camera.open()
    try:
        sim = Simulator(get_preset("zwo-asi224mc", SCALE), clouds=[CloudLayer(cover=1.0, seed=5)], seed=5)
        frame = sim.render(NIGHT)
        path = frame.write(images)
        await camera.comm.send_event(NewImageEvent(f"/cam/{path.name}"))
        log = app._measurement_log.strategy
        assert isinstance(log, MemoryLog)
        await wait_for_points(log, 1)
        assert log.points[0][1]["cloud_total"]["cloudcover"] > 0.7
    finally:
        await camera.close()
        await app.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_influxdb(tmp_path: Path, unused_tcp_port: int) -> None:
    """Optional: writes to a real InfluxDB 2 in Docker (pytest -m integration; needs testcontainers)."""
    influx_module = pytest.importorskip("testcontainers.influxdb2")
    token, org, bucket = "token", "pyobs", "allsky"
    with influx_module.InfluxDb2Container(
        init_mode="setup", admin_token=token, org_name=org, bucket=bucket, username="test", password="5_B2D_Y-Y7hL"
    ) as influx:
        watch = tmp_path / "images"
        watch.mkdir()
        config = module_config("zwo-asi224mc", watch, unused_tcp_port, "cloudcover-influx")
        config["measurement_log"]["logger"] = {
            "type": "influx",
            "url": influx.get_url(),
            "org": org,
            "bucket": bucket,
            "token": token,
        }
        app = Application(**config)
        await app.open()
        try:
            sim = Simulator(get_preset("zwo-asi224mc", SCALE), clouds=[CloudLayer(cover=1.0, seed=5)], seed=5)
            sim.render(NIGHT).write(watch)
            client, _ = influx.get_client(token=token, org_name=org)
            for _ in range(300):
                tables = client.query_api().query(
                    f'from(bucket:"{bucket}") |> range(start: 2026-01-01T00:00:00Z)', org=org
                )
                if tables:
                    break
                await asyncio.sleep(0.1)
            values = {r.get_measurement() + "/" + r.get_field(): r.get_value() for t in tables for r in t.records}
            assert values["cloud_total/cloudcover"] > 0.7
        finally:
            await app.close()


def test_truth_sidecars_are_not_mistaken_for_images(tmp_path: Path) -> None:
    sim = Simulator(get_preset("zwo-asi224mc", 0.1))
    path = sim.render(NIGHT).write(tmp_path)
    assert GroundTruth.path_for(path).exists()
    shutil.copy(path, tmp_path / "copy.fits")
    assert sorted(p.name for p in tmp_path.glob("ccd1_*.fits")) == [path.name]

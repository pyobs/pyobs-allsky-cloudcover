import datetime
from pathlib import Path

import pytest

from pyobs_cloudcover.camera import CameraProfile, Frame, read_image
from pyobs_cloudcover.simulation import CloudLayer, SimulatedFrame, Simulator, get_preset

NIGHT = datetime.datetime(2026, 3, 20, 22, 0, tzinfo=datetime.UTC)
DAY = datetime.datetime(2026, 3, 21, 10, 0, tzinfo=datetime.UTC)
TWILIGHT = datetime.datetime(2026, 3, 20, 17, 45, tzinfo=datetime.UTC)  # sun about -6 deg


class SimulatedCamera:
    """A small simulated ZWO ASI224MC and the camera profile a user would configure for it."""

    def __init__(self, directory: Path, cover: float = 0.4) -> None:
        self.directory = directory
        self.sim = Simulator(get_preset("zwo-asi224mc", 0.4), clouds=[CloudLayer(cover=cover, seed=7)], seed=7)
        self.profile = CameraProfile({"world_model": self.sim.world_model.to_config()})
        self.observer = self.sim._observer

    def frame(self, time: datetime.datetime) -> tuple[Frame, SimulatedFrame]:
        simulated = self.sim.render(time)
        return self.profile.prepare(read_image(simulated.write(self.directory))), simulated


@pytest.fixture
def camera(tmp_path: Path) -> SimulatedCamera:
    return SimulatedCamera(tmp_path)

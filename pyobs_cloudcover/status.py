from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


@dataclass
class ModuleStatus:
    """What the module has been doing, for the /status route."""

    started: datetime.datetime = field(default_factory=_utcnow)
    received: int = 0
    analysed: int = 0
    skipped: int = 0  # a newer image arrived before analysis started
    without_result: int = 0  # no pipeline for the sun altitude, or the pipeline couldn't analyse the image
    failed: int = 0
    last_file: str = ""
    last_obs_time: datetime.datetime | None = None
    last_result_at: datetime.datetime | None = None
    last_pipeline: str = ""
    last_reason: str = ""  # why the last image gave no result
    last_error: str = ""
    last_error_at: datetime.datetime | None = None

    def to_dict(self, now: datetime.datetime | None = None) -> dict[str, Any]:
        now = now or _utcnow()

        def iso(t: datetime.datetime | None) -> str | None:
            return t.isoformat() if t else None

        return {
            "started": iso(self.started),
            "uptime": (now - self.started).total_seconds(),
            "frames": {
                "received": self.received,
                "analysed": self.analysed,
                "skipped": self.skipped,
                "without_result": self.without_result,
                "failed": self.failed,
            },
            "last_file": self.last_file,
            "last_obs_time": iso(self.last_obs_time),
            # age of the result the API currently serves
            "result_age": (now - self.last_obs_time).total_seconds() if self.last_obs_time else None,
            "last_result_at": iso(self.last_result_at),
            "pipeline": self.last_pipeline,
            "reason": self.last_reason,
            "last_error": self.last_error,
            "last_error_at": iso(self.last_error_at),
        }

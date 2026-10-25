"""Validation of the module config (pydantic), so mistakes surface at startup with a clear message.

The camera and watch sections are validated by ``CameraConfig`` and ``WatchConfig``; pipeline options by the
pipelines' own option classes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(default="localhost", description="Interface to listen on, e.g. 0.0.0.0 for all")
    port: int = Field(default=8080, ge=0, le=65535)


class LoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["influx", "memory", "none"] = "influx"
    url: str = ""
    bucket: str = ""
    org: str = ""
    token: str = ""
    timeout: float = Field(default=10.0, gt=0, description="Seconds per write")

    @model_validator(mode="after")
    def _influx_needs_url(self) -> LoggerConfig:
        if self.type == "influx" and not self.url:
            raise ValueError("measurement_log.logger: type 'influx' needs a url (or use type 'none')")
        return self


class FieldConfig(BaseModel):
    """One field of a measurement: the cloud fraction in an area, or the change since the last image."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["total", "zenith", "change", "sun", "moon"]
    name: str
    altitude: float | None = Field(default=None, description="zenith: lower altitude bound (deg)")
    radius: float | None = Field(default=None, description="sun/moon: radius around it (deg)")

    @model_validator(mode="after")
    def _needs(self) -> FieldConfig:
        if self.type == "zenith" and self.altitude is None:
            raise ValueError(f"field {self.name!r} of type zenith needs an altitude")
        if self.type in ("sun", "moon") and self.radius is None:
            raise ValueError(f"field {self.name!r} of type {self.type} needs a radius")
        return self


class MeasurementLogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logger: LoggerConfig = Field(default_factory=lambda: LoggerConfig(type="none"))
    measurements: dict[str, list[FieldConfig]] = Field(default_factory=dict)


class IntervalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float | None = None
    end: float | None = None


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alt_interval: IntervalConfig = Field(default_factory=IntervalConfig)
    options: dict[str, Any] = Field(default_factory=dict)


def validate_pipelines(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    unknown = set(config) - {"day", "night"}
    if unknown:
        raise ValueError(f"Unknown pipeline(s) {sorted(unknown)}; available: day, night")
    return {name: PipelineConfig.model_validate(c or {}).model_dump(exclude_none=False) for name, c in config.items()}

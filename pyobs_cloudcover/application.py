import asyncio
import concurrent.futures
import datetime
import logging
from pathlib import Path
from typing import Any

from pyobs.events import Event, NewImageEvent
from pyobs.modules import Module

from pyobs_cloudcover.camera.profile import CameraProfile, Frame
from pyobs_cloudcover.camera.reader import RawImage, from_fits_hdu, read_image
from pyobs_cloudcover.cloud_coverage_info import CloudCoverageInfo
from pyobs_cloudcover.ingest.watcher import DirectoryWatcher
from pyobs_cloudcover.measurement_log.measurment_logger_factory import MeasurementLoggerFactory
from pyobs_cloudcover.pipeline.pipeline_controller_factory import PipelineControllerFactory
from pyobs_cloudcover.status import ModuleStatus
from pyobs_cloudcover.web_api.server_factory import ServerFactory

log = logging.getLogger(__name__)


class Application(Module):
    """Estimates cloud cover from all-sky images.

    Images come from a directory the capture software writes into (``watch``) and/or as ``NewImageEvent``
    from a pyobs camera module (``image_sender``). ``camera`` describes the camera (see ``CameraConfig``).
    """

    def __init__(
        self,
        server: dict[str, Any],
        measurement_log: dict[str, Any],
        pipelines: dict[str, dict[str, Any]],
        camera: dict[str, Any] | None = None,
        watch: dict[str, Any] | None = None,
        image_sender: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super().__init__(*args, **kwargs)

        if watch is None and image_sender is None:
            raise ValueError("Configure an image source: 'watch' (a directory) and/or 'image_sender' (a camera).")
        self._image_sender = image_sender
        self._camera = CameraProfile(camera)

        server_factory = ServerFactory(self.observer)
        self._server = server_factory(server)

        measurement_logger_factory = MeasurementLoggerFactory(self.observer)
        self._measurement_log = measurement_logger_factory(measurement_log)

        pipeline_controller_factory = PipelineControllerFactory(self.observer, self._camera)
        self._pipeline_controller = pipeline_controller_factory(pipelines)

        # the pipelines keep state between frames, so they run one at a time, off the event loop
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cloudcover")
        self._latest: tuple[str, bool] | None = None
        self._new_image = asyncio.Event()

        self._status = ModuleStatus()
        self._server.set_status_provider(self._status.to_dict)

        self._watcher = DirectoryWatcher(watch, self._submit_file) if watch is not None else None
        if self._watcher is not None:
            self.add_background_task(self._watcher.run)
        self.add_background_task(self._process_images)
        self.add_background_task(self._measurement_log.run)

    async def open(self) -> None:
        await super().open()
        await self._server.start()

        if self._image_sender is not None:
            await self.comm.register_event(NewImageEvent, self.process_new_image)

    async def close(self) -> None:
        await self._server.stop()
        await super().close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def process_new_image(self, event: Event, sender: str) -> bool:
        if not isinstance(event, NewImageEvent):
            return False

        if sender != self._image_sender:
            return False

        self.submit_image(event.filename)
        return True

    def _submit_file(self, path: str) -> None:
        self.submit_image(path, local=True)

    def submit_image(self, filename: str, local: bool = False) -> None:
        """Queue an image (a pyobs VFS path, or a local file if ``local``) for analysis. Only the newest image is
        kept: if analysis falls behind, older images that haven't been started yet are skipped."""
        self._status.received += 1
        if self._latest is not None:
            log.info("Skipping %s, a newer image arrived before it was analysed.", self._latest[0])
            self._status.skipped += 1
            self._finished(*self._latest)
        self._latest = (filename, local)
        self._new_image.set()

    def _finished(self, filename: str, local: bool) -> None:
        if local and self._watcher is not None:
            self._watcher.done(filename)

    async def _process_images(self) -> None:
        while True:
            await self._new_image.wait()
            self._new_image.clear()
            latest, self._latest = self._latest, None
            if latest is not None:
                try:
                    await self.analyse(*latest)
                finally:
                    self._finished(*latest)

    async def _load(self, filename: str, local: bool) -> RawImage:
        if local:
            return await asyncio.to_thread(read_image, filename)
        image = await self.vfs.read_image(filename)
        raw = from_fits_hdu(image.data, image.header)
        raw.path = filename
        return raw

    async def analyse(self, filename: str, local: bool = False) -> bool:
        """Analyse one image; errors are logged, never raised."""
        try:
            log.info("Analysing %s.", filename)
            raw = await self._load(filename, local)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, self._run_pipeline, raw)
        except Exception as e:
            log.exception("Analysing %s failed.", filename)
            self._status.failed += 1
            self._status.last_error = f"{Path(filename).name}: {type(e).__name__}: {e}"
            self._status.last_error_at = datetime.datetime.now(datetime.UTC)
            return False

        self._status.last_file = filename
        if result is None:
            reason = getattr(self._pipeline_controller, "last_reason", "") or "no result"
            log.info("No result for %s: %s.", filename, reason)
            self._status.without_result += 1
            self._status.last_reason = reason
            return False

        measurement, fields = result
        self._server.set_measurement(measurement)
        self._measurement_log.submit(fields, measurement.obs_time)
        self._status.analysed += 1
        self._status.last_obs_time = measurement.obs_time
        self._status.last_result_at = datetime.datetime.now(datetime.UTC)
        self._status.last_pipeline = measurement.pipeline
        self._status.last_reason = ""
        log.info("Finished %s.", filename)
        return True

    def prepare(self, raw: RawImage) -> Frame:
        return self._camera.prepare(raw)

    def _run_pipeline(self, raw: RawImage) -> tuple[CloudCoverageInfo, dict[str, dict[str, float | None]]] | None:
        frame = self.prepare(raw)
        measurement = self._pipeline_controller(frame)
        if measurement is None:
            return None
        return measurement, self._measurement_log.evaluate(measurement)

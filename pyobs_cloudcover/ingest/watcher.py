"""Pick up images that capture software writes into a directory."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

STATE_FILE = ".cloudcover-state.json"


class WatchConfig(BaseModel):
    """The ``watch`` section of the module config."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Directory the capture software writes images into")
    pattern: str = Field(default="*.fits", description="Glob for image files, e.g. '*.fits' or 'image-*.png'")
    recursive: bool = Field(default=False, description="Also watch subdirectories (polling only)")
    poll: bool | None = Field(default=None, description="Poll instead of inotify; default: poll if inotify is missing")
    poll_interval: float = Field(default=2.0, description="Seconds between directory scans when polling")
    settle_time: float = Field(
        default=2.0, description="A polled file must be unchanged for this many seconds before it is read"
    )
    after: Literal["keep", "delete", "move"] = Field(default="keep", description="What to do with handled files")
    archive: str = Field(default="", description="Directory for after: move")
    state_file: str = Field(default="", description=f"Where to remember the last file; default: <path>/{STATE_FILE}")
    process_existing: bool = Field(default=True, description="On start, analyse the newest file not yet analysed")


class DirectoryWatcher:
    """Watches a directory and hands complete new files to ``on_file``, newest first, without a backlog.

    With inotify (Linux, ``asyncinotify`` installed) files are reported once they are closed after writing or
    renamed into the directory. Otherwise the directory is polled and a file is reported once its size and
    modification time haven't changed for ``settle_time``. Hidden files (starting with ``.``) are ignored, so
    capture software that writes a temporary file and renames it is handled either way.

    Only files newer than the last one handled are reported, and if several turn up at once only the newest is;
    the others count as handled (and are deleted/moved if configured). The last handled file is remembered in a
    small state file, so a restart doesn't re-analyse old images.
    """

    def __init__(self, config: WatchConfig | dict[str, object], on_file: Callable[[str], None]) -> None:
        self.config = config if isinstance(config, WatchConfig) else WatchConfig.model_validate(config)
        self._on_file = on_file
        self._dir = Path(self.config.path)
        self._state_path = Path(self.config.state_file) if self.config.state_file else self._dir / STATE_FILE
        self._last_mtime, self._last_name = self._load_state()
        self._seen: dict[str, tuple[int, float, float]] = {}  # path -> (size, mtime, first seen unchanged)
        self._reported: set[str] = set()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> tuple[float, str]:
        try:
            state = json.loads(self._state_path.read_text())
            return float(state["mtime"]), str(state["name"])
        except (OSError, ValueError, KeyError, TypeError):
            return -1.0, ""

    def _save_state(self) -> None:
        try:
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(json.dumps({"mtime": self._last_mtime, "name": self._last_name}))
            os.replace(tmp, self._state_path)
        except OSError as e:
            log.warning("Can't write watcher state %s: %s", self._state_path, e)

    # ------------------------------------------------------------------ files

    def _matches(self, path: Path) -> bool:
        return not path.name.startswith(".") and fnmatch.fnmatch(path.name, self.config.pattern)

    def _candidates(self) -> list[Path]:
        if not self._dir.is_dir():
            return []
        paths = self._dir.rglob("*") if self.config.recursive else self._dir.iterdir()
        return [p for p in paths if p.is_file() and self._matches(p)]

    def _is_new(self, path: Path, mtime: float) -> bool:
        return (mtime, path.name) > (self._last_mtime, self._last_name) and str(path) not in self._reported

    def _report(self, files: list[tuple[float, Path]]) -> None:
        """Report the newest of several complete new files; the older ones count as handled."""
        if not files:
            return
        files.sort(key=lambda f: (f[0], f[1].name))
        for _, older in files[:-1]:
            log.info("Skipping %s, a newer image is already there.", older.name)
            self._reported.add(str(older))
            self.done(str(older))
        mtime, newest = files[-1]
        self._reported.add(str(newest))
        self._last_mtime, self._last_name = max((self._last_mtime, self._last_name), (mtime, newest.name))
        self._save_state()
        self._on_file(str(newest))

    def done(self, filename: str) -> None:
        """Called once a reported (or skipped) file is handled: delete or archive it if configured."""
        self._reported.discard(filename)
        self._seen.pop(filename, None)
        path = Path(filename)
        try:
            if self.config.after == "delete":
                path.unlink(missing_ok=True)
            elif self.config.after == "move" and path.exists():
                target = Path(self.config.archive or self._dir / "archive")
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), target / path.name)
        except OSError as e:
            log.warning("Can't %s %s: %s", self.config.after, filename, e)

    def scan(self, now: float | None = None, settle: bool = True) -> None:
        """One polling pass; ``settle=False`` reports files without waiting (the startup scan)."""
        now = time.time() if now is None else now
        complete: list[tuple[float, Path]] = []
        present = set()
        for path in self._candidates():
            key = str(path)
            present.add(key)
            try:
                st = path.stat()
            except OSError:
                continue
            if not self._is_new(path, st.st_mtime):
                continue
            previous = self._seen.get(key)
            if previous is None or previous[:2] != (st.st_size, st.st_mtime):
                self._seen[key] = (st.st_size, st.st_mtime, now)
                if settle:
                    continue
            elif now - previous[2] < self.config.settle_time:
                continue
            complete.append((st.st_mtime, path))
        for key in list(self._seen):
            if key not in present:
                del self._seen[key]
        self._report(complete)

    def startup(self) -> None:
        if not self.config.process_existing:
            # everything already there counts as handled
            for path in self._candidates():
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                self._last_mtime, self._last_name = max((self._last_mtime, self._last_name), (mtime, path.name))
            return
        self.scan(settle=False)

    # ------------------------------------------------------------------ loops

    async def run(self) -> None:
        if not self._dir.is_dir():
            log.warning("Watch directory %s doesn't exist (yet), waiting for it.", self._dir)
            while not self._dir.is_dir():
                await asyncio.sleep(self.config.poll_interval)
        self.startup()
        use_poll = self.config.poll if self.config.poll is not None else not _has_inotify()
        if use_poll or self.config.recursive:
            await self._run_poll()
        else:
            await self._run_inotify()

    async def _run_poll(self) -> None:
        log.info("Polling %s for %s every %.1f s.", self._dir, self.config.pattern, self.config.poll_interval)
        while True:
            await asyncio.sleep(self.config.poll_interval)
            self.scan()

    async def _run_inotify(self) -> None:
        from asyncinotify import Inotify, Mask

        log.info("Watching %s for %s (inotify).", self._dir, self.config.pattern)
        pending: dict[str, float] = {}
        flush: asyncio.TimerHandle | None = None

        def report() -> None:
            nonlocal flush
            flush = None
            files = [(mtime, Path(p)) for p, mtime in pending.items() if self._is_new(Path(p), mtime)]
            pending.clear()
            self._report(files)

        with Inotify() as inotify:
            inotify.add_watch(self._dir, Mask.CLOSE_WRITE | Mask.MOVED_TO)
            async for event in inotify:
                if event.path is None or not self._matches(Path(event.path)):
                    continue
                try:
                    pending[str(event.path)] = event.path.stat().st_mtime
                except OSError:
                    continue
                # collect files that arrive together, then report only the newest
                if flush is None:
                    flush = asyncio.get_running_loop().call_later(0.2, report)


def _has_inotify() -> bool:
    try:
        import asyncinotify  # noqa: F401
    except (ImportError, OSError):
        return False
    return True

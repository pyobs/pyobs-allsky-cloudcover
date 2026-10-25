import asyncio
import os
from pathlib import Path

import pytest

from pyobs_cloudcover.ingest.watcher import STATE_FILE, DirectoryWatcher


def write(path: Path, mtime: float, content: bytes = b"x") -> Path:
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))
    return path


def make(tmp_path: Path, **config: object) -> tuple[DirectoryWatcher, list[str]]:
    reported: list[str] = []
    watcher = DirectoryWatcher({"path": str(tmp_path), "settle_time": 1.0, **config}, reported.append)
    return watcher, reported


def test_waits_until_a_file_is_complete(tmp_path: Path) -> None:
    watcher, reported = make(tmp_path)
    f = write(tmp_path / "a.fits", 1000)
    watcher.scan(now=0)
    assert reported == []  # first sighting
    f.write_bytes(b"xxxx")  # still being written
    os.utime(f, (1001, 1001))
    watcher.scan(now=0.5)
    watcher.scan(now=1.0)
    assert reported == []  # changed since the last look, settle time restarts
    watcher.scan(now=1.6)
    assert reported == [str(f)]
    watcher.scan(now=5)
    assert reported == [str(f)]  # only once


def test_only_newest_of_several(tmp_path: Path) -> None:
    watcher, reported = make(tmp_path, after="delete")
    for i in range(3):
        write(tmp_path / f"img{i}.fits", 1000 + i)
    watcher.scan(now=0)
    watcher.scan(now=2)
    assert reported == [str(tmp_path / "img2.fits")]
    # the skipped ones count as handled
    assert not (tmp_path / "img0.fits").exists() and not (tmp_path / "img1.fits").exists()


def test_ignores_hidden_and_other_files(tmp_path: Path) -> None:
    watcher, reported = make(tmp_path)
    write(tmp_path / ".tmp123.fits", 1000)
    write(tmp_path / "img.truth.npz", 1000)
    write(tmp_path / "img.png", 1000)
    watcher.scan(now=0)
    watcher.scan(now=2)
    assert reported == []


def test_state_survives_restart(tmp_path: Path) -> None:
    watcher, reported = make(tmp_path)
    write(tmp_path / "a.fits", 1000)
    watcher.startup()
    assert reported == [str(tmp_path / "a.fits")]
    assert (tmp_path / STATE_FILE).exists()

    again, reported_again = make(tmp_path)
    again.startup()
    assert reported_again == []  # already analysed before the restart
    write(tmp_path / "b.fits", 2000)
    again.startup()
    assert reported_again == [str(tmp_path / "b.fits")]


def test_process_existing_false(tmp_path: Path) -> None:
    watcher, reported = make(tmp_path, process_existing=False)
    write(tmp_path / "old.fits", 1000)
    watcher.startup()
    watcher.scan(now=0)
    watcher.scan(now=2)
    assert reported == []


def test_move_after_processing(tmp_path: Path) -> None:
    archive = tmp_path / "done"
    watcher, reported = make(tmp_path, after="move", archive=str(archive))
    write(tmp_path / "a.fits", 1000)
    watcher.startup()
    watcher.done(reported[0])
    assert (archive / "a.fits").exists() and not (tmp_path / "a.fits").exists()


@pytest.mark.asyncio
async def test_run_polling(tmp_path: Path) -> None:
    reported: list[str] = []
    watcher = DirectoryWatcher(
        {"path": str(tmp_path), "poll": True, "poll_interval": 0.05, "settle_time": 0.1}, reported.append
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0.1)
    (tmp_path / "new.fits").write_bytes(b"data")
    for _ in range(40):
        await asyncio.sleep(0.05)
        if reported:
            break
    task.cancel()
    assert reported == [str(tmp_path / "new.fits")]

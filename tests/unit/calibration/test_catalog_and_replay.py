import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.table import Table

from pyobs_cloudcover.calibration.catalog import fetch_hipparcos
from pyobs_cloudcover.calibration.replay import otsu_threshold
from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader


def test_fetch_hipparcos_writes_a_loadable_catalogue(tmp_path: Path) -> None:
    table = Table(
        {"HIP": [1, 2, 3], "RAICRS": [10.0, 20.0, 30.0], "DEICRS": [-5.0, 0.0, 5.0], "Vmag": [6.8, 7.5, np.nan]}
    )
    with patch("astroquery.vizier.VizierClass.query_constraints", return_value=[table]):
        path = fetch_hipparcos(8.0, tmp_path / "hip.csv")
    loader = AltAzCatalogLoader.from_csv(str(path))
    np.testing.assert_array_equal(loader._sao, [1, 2])
    np.testing.assert_allclose(loader._vmag, [6.8, 7.5])


def test_otsu_separates_two_populations() -> None:
    rng = np.random.default_rng(0)
    values = np.r_[rng.normal(3.5, 0.3, 3000), rng.normal(5.8, 0.3, 5000), [np.nan] * 10]
    assert 4.3 < otsu_threshold(values) < 5.0


def test_replay_scores_against_truth(tmp_path: Path) -> None:
    from pyobs_cloudcover.calibration.replay import replay, suggest_thresholds, write_csv
    from pyobs_cloudcover.simulation import CloudLayer, Simulator, get_preset

    paths = []
    for i, cover in enumerate((0.0, 0.5, 1.0)):
        sim = Simulator(get_preset("zwo-asi224mc", 0.4), clouds=[CloudLayer(cover=cover, seed=i)], seed=i)
        paths.append(sim.render(datetime.datetime(2026, 3, 20, 21 + i, tzinfo=datetime.UTC)).write(tmp_path))
    config = {
        "camera": {"world_model": sim.world_model.to_config()},
        "pipelines": {"night": {"alt_interval": {"end": -18}, "options": {"altaz_grid": {"point_number": 20000}}}},
    }
    results = replay(config, paths, sim._observer)
    assert all(r.info is not None and r.truth is not None for r in results)
    (suggestion,) = suggest_thresholds(results)
    assert suggestion.method == "ground truth" and suggestion.agreement is not None
    write_csv(results, tmp_path / "out.csv")
    lines = (tmp_path / "out.csv").read_text().splitlines()
    assert lines[0].startswith("file,time,pipeline") and len(lines) == 4

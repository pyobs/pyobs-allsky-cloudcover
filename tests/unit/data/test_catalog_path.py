import os

from pyobs_cloudcover.data import bundled_catalog_path, resolve_catalog_path
from pyobs_cloudcover.pipeline.night.catalog.altaz_catalog_loader import AltAzCatalogLoader


def test_bundled_catalog_loads() -> None:
    loader = AltAzCatalogLoader.from_csv(bundled_catalog_path())
    assert len(loader._vmag) > 8900


def test_resolve_catalog_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert resolve_catalog_path("") == bundled_catalog_path()
    assert resolve_catalog_path(None) == bundled_catalog_path()
    # 1.x configs pointed at the file in a source checkout
    assert resolve_catalog_path("data/catalog_v50.csv") == bundled_catalog_path()

    own = tmp_path / "my_catalog.csv"
    own.write_text("")
    assert resolve_catalog_path(str(own)) == str(own)
    assert os.path.exists(bundled_catalog_path())

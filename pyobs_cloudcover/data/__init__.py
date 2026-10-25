import logging
import os
from importlib import resources

log = logging.getLogger(__name__)

BUNDLED_CATALOG = "catalog_v50.csv"


def bundled_catalog_path() -> str:
    """Path of the bundled Yale Bright Star Catalogue (V <~ 6.5)."""
    return str(resources.files(__name__).joinpath(BUNDLED_CATALOG))


def resolve_catalog_path(filepath: str | None) -> str:
    """Catalogue file to load. Empty/None means the bundled catalogue.

    Configs written for 1.x point at ``data/catalog_v50.csv`` relative to a source checkout; that file now
    ships inside the package, so such paths fall back to the bundled copy if they don't exist.
    """
    if not filepath:
        return bundled_catalog_path()
    if not os.path.exists(filepath) and os.path.basename(filepath) == BUNDLED_CATALOG:
        log.warning('Catalogue "%s" not found, using the bundled %s instead.', filepath, BUNDLED_CATALOG)
        return bundled_catalog_path()
    return filepath

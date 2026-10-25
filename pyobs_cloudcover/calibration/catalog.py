"""A deeper star catalogue than the bundled Bright Star Catalogue (V <~ 6.5), built on demand from Hipparcos."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def default_catalog_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "pyobs-cloudcover"


def fetch_hipparcos(max_mag: float = 8.0, output: str | Path | None = None) -> Path:
    """Download Hipparcos stars brighter than ``max_mag`` (V) from VizieR (I/239) and write them in the format of
    the bundled catalogue (``;ID;RA;Dec;Vmag``, J2000 degrees), ~25 bytes per star (~1 MB for V < 8).

    Use the file as ``pipelines.night.options.catalog.filepath`` with ``filter.v_mag`` raised accordingly.
    Hipparcos is complete to V ~ 7.3 and covers most stars to ~ 9.
    """
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        raise RuntimeError(
            "Downloading a catalogue needs astroquery: install pyobs-allsky-cloudcover[catalog]"
        ) from None

    output = Path(output) if output else default_catalog_dir() / f"hipparcos_v{max_mag:g}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    vizier = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"], column_filters={"Vmag": f"<{max_mag}"}, row_limit=-1)
    tables = vizier.query_constraints(catalog="I/239/hip_main")
    if not tables:
        raise RuntimeError("VizieR returned nothing")
    t = tables[0]
    write_catalog(output, np.asarray(t["HIP"]), np.asarray(t["RAICRS"]), np.asarray(t["DEICRS"]), np.asarray(t["Vmag"]))
    log.info("Wrote %d stars to %s", len(t), output)
    return output


def write_catalog(path: str | Path, ids: np.ndarray, ra: np.ndarray, dec: np.ndarray, v_mag: np.ndarray) -> None:
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(v_mag)
    with open(path, "w") as f:
        f.write("# Star catalogue for pyobs-cloudcover (Hipparcos, I/239, J2000 degrees)\n")
        f.write("# ;ID;_RAJ2000;_DEJ2000;Vmag\n")
        for i, (n, r, d, v) in enumerate(zip(ids[ok], ra[ok], dec[ok], v_mag[ok])):
            f.write(f"{i};{int(n)};{float(r):.6f};{float(d):.6f};{float(v):.2f}\n")

"""Live control of a running simulator (``pyobs-cloudcover-sim --control PORT``): a small web page and a JSON
endpoint to change the sky between frames. A change triggers the next frame right away."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Conditions:
    """What can be changed while the simulator runs.

    Attributes:
        cover: Cloud cover of the layer, 0..1.
        optical_depth: Cloud extinction in mag (0.3 thin cirrus, >= 3 opaque).
        wind_east, wind_north: Cloud drift in m/s.
        sky_mag: Dark zenith sky brightness in mag/arcsec^2.
        moon: False leaves the moon out.
        time_offset: Hours added to the clock (e.g. -10 to see the previous evening).
    """

    cover: float = 0.3
    optical_depth: float = 3.0
    wind_east: float = 8.0
    wind_north: float = 3.0
    sky_mag: float = 21.3
    moon: bool = True
    time_offset: float = 0.0

    LIMITS = {
        "cover": (0.0, 1.0),
        "optical_depth": (0.0, 10.0),
        "wind_east": (-100.0, 100.0),
        "wind_north": (-100.0, 100.0),
        "sky_mag": (16.0, 23.0),
        "time_offset": (-48.0, 48.0),
    }

    def updated(self, changes: dict[str, Any]) -> Conditions:
        """A copy with ``changes`` applied; unknown keys and out-of-range values raise ``ValueError``."""
        names = {f.name for f in dataclasses.fields(self)}
        values: dict[str, Any] = {}
        for key, value in changes.items():
            if key not in names:
                raise ValueError(f"Unknown setting {key}")
            if key == "moon":
                values[key] = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
                continue
            number = float(value)
            low, high = self.LIMITS[key]
            if not low <= number <= high:
                raise ValueError(f"{key} must be between {low:g} and {high:g}")
            values[key] = number
        return dataclasses.replace(self, **values)


class Control:
    """Thread-safe holder of the current :class:`Conditions`, the last frame's summary and a "changed" event."""

    def __init__(self, conditions: Conditions) -> None:
        self._lock = threading.Lock()
        self._conditions = conditions
        self.last_frame: dict[str, Any] = {}
        self.changed = threading.Event()

    @property
    def conditions(self) -> Conditions:
        with self._lock:
            return self._conditions

    def update(self, changes: dict[str, Any]) -> Conditions:
        with self._lock:
            self._conditions = self._conditions.updated(changes)
            conditions = self._conditions
        log.info("New conditions: %s", conditions)
        self.changed.set()
        return conditions

    def state(self) -> dict[str, Any]:
        return {"conditions": dataclasses.asdict(self.conditions), "last_frame": self.last_frame}

    def serve(self, host: str, port: int) -> ThreadingHTTPServer:
        """Start the web server in a daemon thread."""
        control = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                log.debug(format, *args)

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, data: dict[str, Any]) -> None:
                self._send(status, json.dumps(data).encode(), "application/json")

            def do_GET(self) -> None:
                if self.path.split("?")[0] == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif self.path.split("?")[0] == "/conditions":
                    self._json(200, control.state())
                else:
                    self._json(404, {"error": "not found"})

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_POST(self) -> None:
                if self.path.split("?")[0] != "/conditions":
                    self._json(404, {"error": "not found"})
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
                try:
                    if self.headers.get("Content-Type", "").startswith("application/json"):
                        changes = json.loads(body)
                    else:
                        changes = {k: v[-1] for k, v in parse_qs(body).items()}
                    if not isinstance(changes, dict):
                        raise ValueError("Expected an object")
                    control.update(changes)
                except ValueError as e:
                    self._json(400, {"error": str(e)})
                    return
                self._json(200, control.state())

        server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True, name="sim-control").start()
        log.info("Simulator control on http://%s:%d/", host, server.server_address[1])
        return server


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulated sky</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 12px; font-size: 14px; }
  form { display: grid; grid-template-columns: max-content 1fr 4em; gap: 6px 10px; align-items: center;
         max-width: 640px; }
  input[type=range] { width: 100%; }
  output { text-align: right; font-variant-numeric: tabular-nums; }
  .buttons { grid-column: 1 / -1; display: flex; gap: 8px; align-items: center; }
  #frame { margin-top: 8px; color: #888; font-variant-numeric: tabular-nums; }
  #error { color: #c33; }
</style></head>
<body>
<form id="form">
  <label for="cover">Cloud cover</label>
  <input type="range" id="cover" name="cover" min="0" max="1" step="0.05"><output for="cover"></output>
  <label for="optical_depth">Cloud thickness (mag)</label>
  <input type="range" id="optical_depth" name="optical_depth" min="0" max="5" step="0.1">
  <output for="optical_depth"></output>
  <label for="wind_east">Wind east (m/s)</label>
  <input type="range" id="wind_east" name="wind_east" min="-40" max="40" step="1"><output for="wind_east"></output>
  <label for="wind_north">Wind north (m/s)</label>
  <input type="range" id="wind_north" name="wind_north" min="-40" max="40" step="1"><output for="wind_north"></output>
  <label for="sky_mag">Dark sky (mag/arcsec&sup2;)</label>
  <input type="range" id="sky_mag" name="sky_mag" min="18" max="22" step="0.1"><output for="sky_mag"></output>
  <label for="time_offset">Clock offset (h)</label>
  <input type="range" id="time_offset" name="time_offset" min="-24" max="24" step="0.5">
  <output for="time_offset"></output>
  <label for="moon">Moon</label>
  <span><input type="checkbox" id="moon" name="moon"></span><span></span>
  <div class="buttons"><button type="submit">Apply</button><span id="error"></span></div>
</form>
<div id="frame">&nbsp;</div>
<script>
const form = document.getElementById('form');
const ranges = form.querySelectorAll('input[type=range]');
function show(input) { form.querySelector('output[for=' + input.id + ']').textContent = input.value; }
ranges.forEach(r => r.addEventListener('input', () => show(r)));
function fill(state) {
  const c = state.conditions;
  ranges.forEach(r => { r.value = c[r.id]; show(r); });
  document.getElementById('moon').checked = c.moon;
  frameInfo(state);
}
function frameInfo(state) {
  const f = state.last_frame;
  if (!f.time) return;
  document.getElementById('frame').textContent = 'Last frame ' + f.time.substr(11, 8) + ' UT: sun ' +
    f.sun_alt.toFixed(1) + '\\u00b0, moon ' + f.moon_alt.toFixed(1) + '\\u00b0, true cloud cover (alt > 30\\u00b0) ' +
    (100 * f.cloud_fraction).toFixed(0) + '%';
}
async function load() { fill(await (await fetch('conditions')).json()); }
form.addEventListener('submit', async e => {
  e.preventDefault();
  const data = {};
  ranges.forEach(r => data[r.id] = parseFloat(r.value));
  data.moon = document.getElementById('moon').checked;
  const response = await fetch('conditions', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                              body: JSON.stringify(data)});
  const result = await response.json();
  document.getElementById('error').textContent = response.ok ? 'Applied, next frame is on its way' : result.error;
});
load();
setInterval(async () => frameInfo(await (await fetch('conditions')).json()), 10000);
</script>
</body></html>
"""

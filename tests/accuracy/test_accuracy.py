"""Fails on clear accuracy regressions against tests/accuracy/baseline.json (run: pytest -m accuracy).

Regenerate the baseline with ``benchmarks/accuracy.py --baseline`` after a deliberate change, and say why in
the commit message.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

# how much worse than the baseline a camera/pipeline may get before the test fails
MAX_AGREEMENT_DROP = 0.05
MAX_TOTAL_ERROR_RISE = 0.05
# agreement only counts the sky a result covers, so coverage mustn't shrink either
MAX_COVERAGE_DROP = 0.05


@pytest.mark.accuracy
def test_no_accuracy_regression() -> None:
    import accuracy  # benchmarks/accuracy.py

    baseline = json.loads((ROOT / "tests" / "accuracy" / "baseline.json").read_text())
    summary = accuracy.summarise(accuracy.run(baseline["scale"], baseline["seeds"]))

    problems = []
    for key, base in baseline["summary"].items():
        now = summary.get(key)
        if now is None:
            problems.append(f"{key}: no results any more")
            continue
        if now["agreement"] < base["agreement"] - MAX_AGREEMENT_DROP:
            problems.append(f"{key}: agreement {now['agreement']:.3f} < baseline {base['agreement']:.3f}")
        if now["total_error"] > base["total_error"] + MAX_TOTAL_ERROR_RISE:
            problems.append(f"{key}: total error {now['total_error']:.3f} > baseline {base['total_error']:.3f}")
        if now.get("coverage", 1.0) < base.get("coverage", 1.0) - MAX_COVERAGE_DROP:
            problems.append(f"{key}: coverage {now['coverage']:.3f} < baseline {base['coverage']:.3f}")
    assert not problems, "\n".join(problems)

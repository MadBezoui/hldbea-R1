import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "analysis" / "manuscript_assets.py"
    spec = importlib.util.spec_from_file_location("manuscript_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runs_and_report():
    from hldbea.statistics import holm_step_down, observations_digest, paired_wilcoxon

    rng = np.random.default_rng(4)
    runs = []
    for label, shift in (("a:x", 0.0), ("b:y", 0.3)):
        for seed in range(12):
            value = float(rng.normal(1.0 + shift, 0.1))
            runs.append({"run_id": f"{label}-{seed}", "problem": "p", "label": label,
                         "seed": seed, "values": {"hv": value}})
    first = {r["seed"]: r["values"]["hv"] for r in runs if r["label"] == "a:x"}
    second = {r["seed"]: r["values"]["hv"] for r in runs if r["label"] == "b:y"}
    p = paired_wilcoxon(first, second, direction="higher").p_value
    holm = holm_step_down({"h": p}, family="final-hv")["h"]
    from dataclasses import asdict

    from hldbea.statistics import matched_pairs_rank_biserial, vargha_delaney_a12

    ordered = sorted(first)
    effect = vargha_delaney_a12([first[s] for s in ordered], [second[s] for s in ordered],
                                direction="higher")
    paired = matched_pairs_rank_biserial([first[s] for s in ordered],
                                         [second[s] for s in ordered], direction="higher")
    report = {
        "metric_directions": {"hv": "higher"},
        "alpha": 0.05,
        "observations_digest": observations_digest(
            (r["run_id"], r["problem"], r["label"], r["seed"], r["values"]) for r in runs
        ),
        "summaries": [
            {"problem_id": "p", "algorithm": lab, "metric": "hv", "n": 12,
             "median": float(np.median([r["values"]["hv"] for r in runs if r["label"] == lab]))}
            for lab in ("a:x", "b:y")
        ],
        "paired_tests": [{"problem_id": "p", "reference": "a:x", "competitor": "b:y",
                          "metric": "hv", "direction": "higher", "family": "final-hv",
                          "hypothesis": "h", "p_raw": p, "reject_holm": holm.reject,
                          "effect_size": asdict(effect), "paired_rank_biserial": paired}],
    }
    return runs, report


def test_matching_report_is_accepted():
    module = _module()
    runs, report = _runs_and_report()
    module.verify_statistics(runs, report, "report")


@pytest.mark.parametrize(
    "tamper", ["median", "decision", "observation", "interpretation", "a12", "paired"]
)
def test_report_of_same_size_but_different_content_is_rejected(tamper):
    module = _module()
    runs, report = _runs_and_report()
    if tamper == "median":
        report["summaries"][0]["median"] = 123456789.0
    elif tamper == "decision":
        report["paired_tests"][0]["reject_holm"] = not report["paired_tests"][0]["reject_holm"]
    elif tamper == "interpretation":
        # Only the field that sets the sign of a published comparison is altered.
        effect = report["paired_tests"][0]["effect_size"]
        effect["interpretation"] = (
            "first_better" if effect["interpretation"] != "first_better" else "second_better"
        )
    elif tamper == "a12":
        report["paired_tests"][0]["effect_size"]["a12"] = 0.5
    elif tamper == "paired":
        report["paired_tests"][0]["paired_rank_biserial"] *= -1
    else:
        runs[0]["values"]["hv"] += 1e-6
    with pytest.raises(module.AssetError):
        module.verify_statistics(runs, report, "report")


def test_matched_pairs_rank_biserial_sign_and_range():
    from hldbea.statistics import matched_pairs_rank_biserial

    assert matched_pairs_rank_biserial([2, 3, 4], [1, 1, 1], direction="higher") == 1.0
    assert matched_pairs_rank_biserial([2, 3, 4], [1, 1, 1], direction="lower") == -1.0
    assert matched_pairs_rank_biserial([1, 1], [1, 1], direction="higher") == 0.0

#!/usr/bin/env python3
"""
Wave 5B.4 / 5C gate report utility.

Reads run artifacts and prints:
- peak validation metrics (overall + low band)
- overfit severity (peak - last val)
- test overall + low/mid/high SNR-bin averages
- decision-gate pass/fail checks for W5B.3/W5B.4 and SSL promotion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


LOW_20_6 = [-20, -18, -16, -14, -12, -10, -8, -6]
LOW_14_6 = [-14, -12, -10, -8, -6]
MID_4_4 = [-4, -2, 0, 2, 4]
HIGH_6_18 = [6, 8, 10, 12, 14, 16, 18]


@dataclass
class RunMetrics:
    name: str
    run_dir: Path
    peak_epoch_1b: int
    peak_val_acc: float
    peak_val_low_macro_acc: float
    peak_train_acc: float
    last_epoch_1b: int
    last_val_acc: float
    last_train_acc: float
    overfit_drop: float
    test_acc: Optional[float]
    test_low_20_6: Optional[float]
    test_low_14_6: Optional[float]
    test_mid_4_4: Optional[float]
    test_high_6_18: Optional[float]


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _avg_snr_bins(by_snr: Dict[str, float], snrs: Iterable[int]) -> Optional[float]:
    vals = [float(by_snr[str(s)]) for s in snrs if str(s) in by_snr]
    if not vals:
        return None
    return float(mean(vals))


def read_run(run_dir: Path) -> RunMetrics:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}")

    rows = _load_jsonl(metrics_path)
    val_rows = [r for r in rows if r.get("val_acc") is not None]
    if not val_rows:
        raise ValueError(f"no val rows in {metrics_path}")

    peak = max(val_rows, key=lambda r: float(r["val_acc"]))
    last = val_rows[-1]
    overfit_drop = float(peak["val_acc"]) - float(last["val_acc"])

    # Test accuracy is written to the final summary record in metrics.jsonl
    test_acc: Optional[float] = None
    for r in reversed(rows):
        if r.get("test_acc") is not None:
            test_acc = float(r["test_acc"])
            break

    by_snr_path = run_dir / "test_acc_by_snr.json"
    test_low_20_6: Optional[float] = None
    test_low_14_6: Optional[float] = None
    test_mid_4_4: Optional[float] = None
    test_high_6_18: Optional[float] = None
    if by_snr_path.exists():
        by_snr = json.loads(by_snr_path.read_text(encoding="utf-8"))
        test_low_20_6 = _avg_snr_bins(by_snr, LOW_20_6)
        test_low_14_6 = _avg_snr_bins(by_snr, LOW_14_6)
        test_mid_4_4 = _avg_snr_bins(by_snr, MID_4_4)
        test_high_6_18 = _avg_snr_bins(by_snr, HIGH_6_18)

    return RunMetrics(
        name=run_dir.name,
        run_dir=run_dir,
        peak_epoch_1b=int(peak["epoch"]) + 1,
        peak_val_acc=float(peak["val_acc"]),
        peak_val_low_macro_acc=float(peak.get("val_low_macro_acc", 0.0)),
        peak_train_acc=float(peak.get("train_acc", 0.0)),
        last_epoch_1b=int(last["epoch"]) + 1,
        last_val_acc=float(last["val_acc"]),
        last_train_acc=float(last.get("train_acc", 0.0)),
        overfit_drop=overfit_drop,
        test_acc=test_acc,
        test_low_20_6=test_low_20_6,
        test_low_14_6=test_low_14_6,
        test_mid_4_4=test_mid_4_4,
        test_high_6_18=test_high_6_18,
    )


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


def _delta(a: float, b: float) -> float:
    return float(a - b)


def _gate_line(name: str, passed: bool, detail: str) -> str:
    status = "PASS" if passed else "FAIL"
    return f"[{status}] {name}: {detail}"


def evaluate_w5b3_gates(run_map: Dict[str, RunMetrics]) -> List[str]:
    lines: List[str] = []
    needed = ["D0", "D1", "D2", "D3", "D4", "D5"]
    if any(k not in run_map for k in needed):
        missing = [k for k in needed if k not in run_map]
        lines.append(f"[WARN] missing W5B.3 runs for gate checks: {', '.join(missing)}")
        return lines

    d0 = run_map["D0"]
    d1 = run_map["D1"]
    d2 = run_map["D2"]
    d3 = run_map["D3"]
    d4 = run_map["D4"]
    d5 = run_map["D5"]

    d2_d1_low = _delta(d2.peak_val_low_macro_acc, d1.peak_val_low_macro_acc)
    d2_d1_all = _delta(d2.peak_val_acc, d1.peak_val_acc)
    pass_d2_d1 = (d2_d1_low >= 0.010) or (d2_d1_all >= 0.003)
    lines.append(
        _gate_line(
            "D2 vs D1",
            pass_d2_d1,
            f"delta_low={d2_d1_low:+.4f}, delta_overall={d2_d1_all:+.4f}, criterion=(low>=+0.010 or overall>=+0.003)",
        )
    )

    d3_d2_low = _delta(d3.peak_val_low_macro_acc, d2.peak_val_low_macro_acc)
    d3_d2_all = _delta(d3.peak_val_acc, d2.peak_val_acc)
    pass_d3_d2 = d3_d2_low >= 0.010
    lines.append(
        _gate_line(
            "D3 vs D2",
            pass_d3_d2,
            f"delta_low={d3_d2_low:+.4f}, delta_overall={d3_d2_all:+.4f}, criterion=(low>=+0.010)",
        )
    )

    for tag, run in (("D4", d4), ("D5", d5)):
        d_all = _delta(run.peak_val_acc, d0.peak_val_acc)
        d_low = _delta(run.peak_val_low_macro_acc, d0.peak_val_low_macro_acc)
        high0 = d0.test_high_6_18
        highx = run.test_high_6_18
        d_high = None if (high0 is None or highx is None) else _delta(highx, high0)
        pass_dep = (
            d_all >= 0.003
            and d_low >= 0.010
            and (d_high is not None and d_high >= -0.003)
        )
        lines.append(
            _gate_line(
                f"{tag} vs D0 (deployable)",
                pass_dep,
                f"delta_overall={d_all:+.4f}, delta_low={d_low:+.4f}, delta_high={_fmt(d_high)} (need overall>=+0.003, low>=+0.010, high>=-0.003)",
            )
        )
    return lines


def evaluate_w5b4_stage1_gates(run_map: Dict[str, RunMetrics]) -> List[str]:
    lines: List[str] = []
    needed = ["D0", "D2", "C0", "C1"]
    if any(k not in run_map for k in needed):
        missing = [k for k in needed if k not in run_map]
        lines.append(f"[WARN] missing W5B.4 Stage-1 runs for gate checks: {', '.join(missing)}")
        return lines

    d0 = run_map["D0"]
    d2 = run_map["D2"]
    c0 = run_map["C0"]
    c1 = run_map["C1"]

    c0_all = _delta(c0.peak_val_acc, d2.peak_val_acc)
    c0_low = _delta(c0.peak_val_low_macro_acc, d2.peak_val_low_macro_acc)
    pass_c0 = (c0_all >= 0.003) or (c0_low >= 0.010)
    lines.append(
        _gate_line(
            "C0 vs D2",
            pass_c0,
            f"delta_overall={c0_all:+.4f}, delta_low={c0_low:+.4f}, criterion=(overall>=+0.003 or low>=+0.010)",
        )
    )

    c1_all = _delta(c1.peak_val_acc, d0.peak_val_acc)
    c1_low = _delta(c1.peak_val_low_macro_acc, d0.peak_val_low_macro_acc)
    c1_high = None
    if d0.test_high_6_18 is not None and c1.test_high_6_18 is not None:
        c1_high = _delta(c1.test_high_6_18, d0.test_high_6_18)
    pass_c1 = (
        c1_all >= 0.003
        and c1_low >= 0.010
        and (c1_high is not None and c1_high >= -0.003)
    )
    lines.append(
        _gate_line(
            "C1 vs D0",
            pass_c1,
            f"delta_overall={c1_all:+.4f}, delta_low={c1_low:+.4f}, delta_high={_fmt(c1_high)} (need overall>=+0.003, low>=+0.010, high>=-0.003)",
        )
    )

    return lines


def evaluate_ssl_gate(run_map: Dict[str, RunMetrics], anchor_floor: float = 0.6409) -> List[str]:
    lines: List[str] = []
    ssl_keys = [k for k in run_map.keys() if k.startswith("S")]
    if not ssl_keys:
        lines.append("[WARN] no S* runs mapped; skipping SSL promotion gate")
        return lines

    if "S0" not in run_map:
        lines.append("[WARN] S0 not mapped; skipping SSL promotion gate")
        return lines

    s0 = run_map["S0"]
    baseline = max(float(s0.test_acc or 0.0), float(anchor_floor))
    target = baseline + 0.008

    best: Optional[Tuple[str, RunMetrics]] = None
    for key in sorted(ssl_keys):
        rm = run_map[key]
        if rm.test_acc is None:
            continue
        if best is None or float(rm.test_acc) > float(best[1].test_acc or -1.0):
            best = (key, rm)

    if best is None:
        lines.append("[WARN] no SSL test_acc values found")
        return lines

    key, run = best
    low_gain = None
    high_drop = None
    if run.test_low_20_6 is not None and s0.test_low_20_6 is not None:
        low_gain = run.test_low_20_6 - s0.test_low_20_6
    if run.test_high_6_18 is not None and s0.test_high_6_18 is not None:
        high_drop = run.test_high_6_18 - s0.test_high_6_18

    passed = (
        run.test_acc is not None
        and run.test_acc >= target
        and low_gain is not None
        and low_gain >= 0.015
        and high_drop is not None
        and high_drop >= -0.003
    )
    lines.append(
        _gate_line(
            "SSL promotion",
            passed,
            f"best={key} test_acc={_fmt(run.test_acc)} target={target:.4f}, low_gain={_fmt(low_gain)}, high_delta={_fmt(high_drop)}",
        )
    )
    return lines


def print_table(rows: List[RunMetrics]) -> None:
    headers = [
        "run",
        "peak_val",
        "peak_low(-14..-6)",
        "peak_ep",
        "last_val",
        "overfit_drop",
        "test_acc",
        "test_low(-20..-6)",
        "test_mid(-4..+4)",
        "test_high(+6..+18)",
    ]
    print("\t".join(headers))
    for r in rows:
        print(
            "\t".join(
                [
                    r.name,
                    _fmt(r.peak_val_acc),
                    _fmt(r.peak_val_low_macro_acc),
                    str(r.peak_epoch_1b),
                    _fmt(r.last_val_acc),
                    _fmt(r.overfit_drop),
                    _fmt(r.test_acc),
                    _fmt(r.test_low_20_6),
                    _fmt(r.test_mid_4_4),
                    _fmt(r.test_high_6_18),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 5B.4/5C gate report")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Run directory path (repeat). If omitted, defaults to w5b3_* in athena/goose roots.",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Symbolic mapping like D0=/abs/path/to/run_dir. Used for gate checks.",
    )
    parser.add_argument(
        "--anchor-floor",
        type=float,
        default=0.6409,
        help="External baseline floor used in SSL gate max(S0, anchor_floor)+0.008",
    )
    parser.add_argument("--skip-w5b3-gates", action="store_true")
    parser.add_argument("--skip-w5b4-gates", action="store_true")
    parser.add_argument("--skip-ssl-gate", action="store_true")
    args = parser.parse_args()

    run_dirs: List[Path] = []
    if args.run_dir:
        run_dirs = [Path(p).resolve() for p in args.run_dir]
    else:
        default_roots = [
            Path("runs/rml2016_athena"),
            Path("runs/rml2016_goose"),
        ]
        for root in default_roots:
            if not root.exists():
                continue
            run_dirs.extend(sorted(root.glob("w5b3_*")))

    if not run_dirs:
        raise SystemExit("no run dirs found; pass --run-dir explicitly")

    rows: List[RunMetrics] = []
    for rd in run_dirs:
        try:
            rows.append(read_run(rd))
        except Exception as e:  # pragma: no cover - operator visibility is more important than strict failure
            print(f"[WARN] skipping {rd}: {e}")

    if not rows:
        raise SystemExit("no readable runs")

    rows = sorted(rows, key=lambda r: r.name)
    print_table(rows)

    sym_map: Dict[str, RunMetrics] = {}
    for m in args.map:
        if "=" not in m:
            raise SystemExit(f"invalid --map format: {m}")
        key, path = m.split("=", 1)
        key = key.strip()
        path_obj = Path(path.strip()).resolve()
        match = next((r for r in rows if r.run_dir.resolve() == path_obj), None)
        if match is None:
            try:
                match = read_run(path_obj)
            except Exception as e:  # pragma: no cover - operator visibility is more important than strict failure
                print(f"[WARN] skipping map {key}={path_obj}: {e}")
                continue
        sym_map[key] = match

    if sym_map:
        print("\n# Gate checks")
        if not args.skip_w5b3_gates:
            print("## W5B.3")
            for line in evaluate_w5b3_gates(sym_map):
                print(line)
        if not args.skip_w5b4_gates:
            print("## W5B.4 Stage-1")
            for line in evaluate_w5b4_stage1_gates(sym_map):
                print(line)
        if not args.skip_ssl_gate:
            print("## SSL promotion")
            for line in evaluate_ssl_gate(sym_map, anchor_floor=float(args.anchor_floor)):
                print(line)


if __name__ == "__main__":
    main()

import argparse
import json
import shlex
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_value(value: str):
    value = value.strip()
    if value.lower() in {"none", "null"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_grid(grid: str) -> Dict[str, List]:
    if not grid:
        return {}
    result: Dict[str, List] = {}
    for chunk in grid.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid grid chunk: {chunk}")
        key, values = chunk.split("=", 1)
        vals = [parse_value(v) for v in values.split(",") if v.strip() != ""]
        if not vals:
            raise ValueError(f"No values provided for grid key: {key}")
        result[key.strip()] = vals
    return result


def format_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        txt = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        txt = str(value)
    txt = txt.replace("-", "m").replace(".", "p")
    return txt


def build_run_name(params: Dict[str, object]) -> str:
    parts = []
    for key, value in params.items():
        parts.append(f"{key}{format_value(value)}")
    return "_".join(parts)


def iter_param_sets(grid: Dict[str, List]) -> Iterable[Dict[str, object]]:
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def read_best_val(metrics_path: Path) -> Tuple[float, float]:
    best_val = None
    last_val = None
    if not metrics_path.exists():
        return None, None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "val_acc" in record:
                last_val = record["val_acc"]
                if best_val is None or record["val_acc"] > best_val:
                    best_val = record["val_acc"]
    return best_val, last_val


def write_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep runner for DiT-AMC.")
    parser.add_argument("--train-script", type=str, default="train.py")
    parser.add_argument("--out-dir-root", type=str, default="./runs/sweep")
    parser.add_argument(
        "--grid",
        type=str,
        required=True,
        help="Grid definition, e.g. 't_max=100,200;p_clean=0.1,0.3;lambda_diff=0.1,0.2'",
    )
    parser.add_argument(
        "--base-args",
        type=str,
        default="",
        help="Extra args passed to train.py (e.g. '--data-path ... --preset B --snr-mode predict --amp').",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = parse_grid(args.grid)
    out_root = Path(args.out_dir_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "sweep_results.jsonl"

    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = (Path(__file__).resolve().parent / train_script).resolve()
    train_cwd = str(train_script.parent)

    runs = list(iter_param_sets(grid))
    if args.max_runs and args.max_runs > 0:
        runs = runs[: args.max_runs]

    base_args = shlex.split(args.base_args)

    for idx, params in enumerate(runs, start=1):
        run_name = build_run_name(params)
        run_dir = out_root / run_name
        metrics_path = run_dir / "metrics.jsonl"

        if args.skip_existing and metrics_path.exists():
            best_val, last_val = read_best_val(metrics_path)
            record = {
                "run_dir": str(run_dir),
                "params": params,
                "best_val_acc": best_val,
                "last_val_acc": last_val,
                "status": "skipped",
            }
            write_jsonl(results_path, record)
            continue

        cmd = [sys.executable, str(train_script), "--out-dir", str(run_dir)]
        cmd.extend(base_args)
        for key, value in params.items():
            arg_key = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{arg_key}")
            elif value is None:
                continue
            else:
                cmd.extend([f"--{arg_key}", str(value)])

        if args.dry_run:
            print(" ".join(cmd))
            continue

        start = time.time()
        try:
            subprocess.run(cmd, check=True, cwd=train_cwd)
            status = "ok"
        except subprocess.CalledProcessError as exc:
            status = f"failed({exc.returncode})"
            if not args.keep_going:
                raise
        elapsed = time.time() - start

        best_val, last_val = read_best_val(metrics_path)
        record = {
            "run_dir": str(run_dir),
            "params": params,
            "best_val_acc": best_val,
            "last_val_acc": last_val,
            "status": status,
            "elapsed_sec": elapsed,
        }
        write_jsonl(results_path, record)


if __name__ == "__main__":
    main()

import argparse
import json
import os
from typing import Dict, List, Tuple


def _read_acc_by_snr(path: str) -> Dict[int, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, float] = {}
    for k, v in data.items():
        out[int(k)] = float(v)
    return out


def _sorted_xy(acc_by_snr: Dict[int, float]) -> Tuple[List[int], List[float]]:
    xs = sorted(acc_by_snr.keys())
    ys = [acc_by_snr[x] for x in xs]
    return xs, ys


def _default_label(path: str) -> str:
    # If file is ".../runs/<run_name>/test_acc_by_snr.json", use "<run_name>".
    d = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot accuracy vs SNR from *_acc_by_snr.json.")
    p.add_argument(
        "--json",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to JSON files like test_acc_by_snr.json",
    )
    p.add_argument(
        "--label",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels (same count/order as --json). Defaults to parent directory name.",
    )
    p.add_argument("--title", type=str, default="Accuracy vs SNR")
    p.add_argument("--out", type=str, required=True, help="Output path (.png or .pdf).")
    p.add_argument("--show", action="store_true", help="Show interactive window (if available).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.label is not None and len(args.label) not in (0, len(args.json)):
        raise ValueError("--label must be omitted or have same length as --json")
    labels = args.label if args.label else [_default_label(p) for p in args.json]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.5, 4.8), dpi=140)
    for path, label in zip(args.json, labels):
        acc = _read_acc_by_snr(path)
        xs, ys = _sorted_xy(acc)
        plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.0, label=label)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(args.title)
    if len(args.json) > 1:
        plt.legend(loc="lower right")
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


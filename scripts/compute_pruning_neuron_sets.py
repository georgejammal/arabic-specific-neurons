from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ACT_DIR_DEFAULT = ROOT / "eval" / "activations"
OUT_DIR_DEFAULT = ROOT / "eval" / "plots" / "neuron_sets"
ABLATION_DIR_DEFAULT = ROOT / "eval" / "ablation_sets"


def load_activations(act_dir: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Load activation pairs from act_dir, skipping any file with non-finite values.
    Returns EN, AR tensors shaped [N, L, M] and the list of skipped basenames.
    """
    paths = sorted(act_dir.glob("*.pt"))
    if not paths:
        raise SystemExit(f"No .pt files found in {act_dir}")

    en_list: list[torch.Tensor] = []
    ar_list: list[torch.Tensor] = []
    skipped: list[str] = []

    for p in paths:
        d = torch.load(p, map_location="cpu")
        en = d["en_mlp_gated_lasttok"]
        ar = d["ar_mlp_gated_lasttok"]
        if (not torch.isfinite(en).all()) or (not torch.isfinite(ar).all()):
            skipped.append(p.name)
            continue
        en_list.append(en.float())
        ar_list.append(ar.float())

    if not en_list:
        raise SystemExit(f"All activation files were non-finite in {act_dir}")

    EN = torch.stack(en_list, dim=0)  # [N,L,M]
    AR = torch.stack(ar_list, dim=0)
    return EN, AR, skipped


def per_layer_counts(layers: Iterable[int], n_layers: int) -> list[int]:
    counts = [0] * n_layers
    for l in layers:
        counts[l] += 1
    return counts


def save_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 60):
    plt.figure(figsize=(6, 3))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_layer_counts(counts: list[int], out_path: Path, title: str):
    plt.figure(figsize=(10, 3))
    plt.bar(list(range(len(counts))), counts)
    plt.title(title)
    plt.xlabel("layer")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_set(
    *,
    name: str,
    mask: np.ndarray,  # [L,M] bool
    delta_signed: np.ndarray,
    mu_ar_abs: np.ndarray,
    mu_en_abs: np.ndarray,
    si: np.ndarray,
    denom: np.ndarray,
    out_dir: Path,
    ablation_dir: Path,
    metric_label: str,
    bins: int,
):
    L, M = mask.shape
    layers, neurons = np.nonzero(mask)
    rows = []
    for l, n in zip(layers.tolist(), neurons.tolist()):
        rows.append(
            {
                "layer": int(l),
                "neuron": int(n),
                "delta_mean_signed": float(delta_signed[l, n]),
                "mean_abs_ar": float(mu_ar_abs[l, n]),
                "mean_abs_en": float(mu_en_abs[l, n]),
                "si": float(si[l, n]),
                "denom": float(denom[l, n]),
            }
        )

    json_path = ablation_dir / f"{name}.json"
    tsv_path = ablation_dir / f"{name}.tsv"
    hist_path = out_dir / f"{name}_{metric_label}_hist.png"
    layer_path = out_dir / f"{name}_layer_counts.png"

    payload = {
        "name": name,
        "count": int(mask.sum()),
        "rows": rows,
        "metric": metric_label,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # TSV with stable columns
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("layer\tneuron\tdelta_mean_signed\tmean_abs_ar\tmean_abs_en\tsi\tdenom\n")
        for r in rows:
            f.write(
                f"{r['layer']}\t{r['neuron']}\t{r['delta_mean_signed']}\t"
                f"{r['mean_abs_ar']}\t{r['mean_abs_en']}\t{r['si']}\t{r['denom']}\n"
            )

    vals = delta_signed[mask] if "delta" in metric_label else si[mask]
    save_hist(
        vals,
        hist_path,
        title=f"{name} ({metric_label}), n={vals.size}",
        xlabel=metric_label,
        bins=bins,
    )
    save_layer_counts(
        per_layer_counts(layers.tolist(), n_layers=L),
        layer_path,
        title=f"{name} per-layer counts",
    )

    return {
        "name": name,
        "count": int(mask.sum()),
        "json": str(json_path),
        "tsv": str(tsv_path),
        "hist": str(hist_path),
        "layer_counts": str(layer_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute neuron sets for pruning experiments (delta, SI, intersection).")
    ap.add_argument("--act-dir", type=Path, default=ACT_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--ablation-dir", type=Path, default=ABLATION_DIR_DEFAULT)
    ap.add_argument(
        "--set-tag",
        type=str,
        default="",
        help="Optional tag appended after set name (e.g., '_2' -> setA_2_delta_gt_...).",
    )
    ap.add_argument("--delta-thr", type=float, default=1.2, help="Threshold for signed mean delta (ar - en).")
    ap.add_argument("--si-thr", type=float, default=0.85, help="Threshold for SI on mean(|act|).")
    ap.add_argument("--si-min-denom", type=float, default=0.1, help="Minimum (mu_ar + mu_en) for SI selection.")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ablation_dir.mkdir(parents=True, exist_ok=True)

    EN, AR, skipped = load_activations(args.act_dir)
    used = EN.shape[0]
    L, M = EN.shape[1], EN.shape[2]

    en_mean = EN.mean(dim=0).numpy()  # signed mean
    ar_mean = AR.mean(dim=0).numpy()
    delta_signed = ar_mean - en_mean  # [L,M]

    mu_en_abs = EN.abs().mean(dim=0).numpy()
    mu_ar_abs = AR.abs().mean(dim=0).numpy()
    denom = mu_ar_abs + mu_en_abs
    si = (mu_ar_abs - mu_en_abs) / (denom + float(args.eps))

    mask_a = delta_signed > float(args.delta_thr)
    mask_b = (si > float(args.si_thr)) & (denom > float(args.si_min_denom))
    mask_c = mask_a & mask_b

    summary = {
        "used_activation_files": used,
        "skipped_activation_files": skipped,
        "shape": [L, M],
        "thresholds": {
            "delta_thr": float(args.delta_thr),
            "si_thr": float(args.si_thr),
            "si_min_denom": float(args.si_min_denom),
            "eps": float(args.eps),
        },
        "sets": [],
    }

    summary["sets"].append(
        export_set(
            name=f"setA{args.set_tag}_delta_gt_{args.delta_thr}",
            mask=mask_a,
            delta_signed=delta_signed,
            mu_ar_abs=mu_ar_abs,
            mu_en_abs=mu_en_abs,
            si=si,
            denom=denom,
            out_dir=args.out_dir,
            ablation_dir=args.ablation_dir,
            metric_label="delta_signed",
            bins=args.bins,
        )
    )
    summary["sets"].append(
        export_set(
            name=f"setB{args.set_tag}_si_gt_{args.si_thr}_den_gt_{args.si_min_denom}",
            mask=mask_b,
            delta_signed=delta_signed,
            mu_ar_abs=mu_ar_abs,
            mu_en_abs=mu_en_abs,
            si=si,
            denom=denom,
            out_dir=args.out_dir,
            ablation_dir=args.ablation_dir,
            metric_label="si",
            bins=args.bins,
        )
    )
    summary["sets"].append(
        export_set(
            name=f"setC{args.set_tag}_intersection_delta_and_si",
            mask=mask_c,
            delta_signed=delta_signed,
            mu_ar_abs=mu_ar_abs,
            mu_en_abs=mu_en_abs,
            si=si,
            denom=denom,
            out_dir=args.out_dir,
            ablation_dir=args.ablation_dir,
            metric_label="delta_signed",
            bins=args.bins,
        )
    )

    summary_path = args.out_dir / "summary_pruning_sets.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done. Wrote summary to:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

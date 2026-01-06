from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MEAN_JSON_DEFAULT = ROOT / "eval" / "mean_mlp_gated_lasttok_en_ar.json"
ACT_DIR_DEFAULT = ROOT / "eval" / "activations"
OUT_DIR_DEFAULT = ROOT / "eval" / "plots" / "neuron_sets"


def _ensure_matplotlib():
    import matplotlib.pyplot as plt  # noqa: F401


def save_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 60):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_layer_counts(counts: list[int], out_path: Path, title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 3))
    plt.bar(list(range(len(counts))), counts)
    plt.title(title)
    plt.xlabel("layer")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def per_layer_counts(layers: list[int], n_layers: int) -> list[int]:
    c = Counter(layers)
    return [c.get(i, 0) for i in range(n_layers)]


def compute_abs_means_from_pt(act_dir: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    act_paths = sorted(act_dir.glob("*.pt"))
    if not act_paths:
        raise SystemExit(f"No activation .pt files found in {act_dir}")

    sum_en = None
    sum_ar = None
    used = 0
    skipped = 0

    for p in act_paths:
        d = torch.load(p, map_location="cpu")
        en = d["en_mlp_gated_lasttok"]
        ar = d["ar_mlp_gated_lasttok"]
        if not torch.isfinite(en).all() or not torch.isfinite(ar).all():
            skipped += 1
            continue

        en = en.float().abs()
        ar = ar.float().abs()

        if sum_en is None:
            sum_en = torch.zeros_like(en)
            sum_ar = torch.zeros_like(ar)

        sum_en += en
        sum_ar += ar
        used += 1

    if used == 0:
        raise SystemExit("All activation files were non-finite; cannot compute mean(|act|).")

    mu_en = (sum_en / used).numpy()
    mu_ar = (sum_ar / used).numpy()
    return mu_en, mu_ar, used, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-json", type=Path, default=MEAN_JSON_DEFAULT)
    ap.add_argument("--act-dir", type=Path, default=ACT_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)

    ap.add_argument("--delta-thr", type=float, default=1.2, help="Threshold for delta-based neuron sets.")
    ap.add_argument("--si-thr", type=float, default=0.85, help="Threshold for SI-based neuron set (Arabic specialists).")
    ap.add_argument("--si-min-denom", type=float, default=0.1, help="Min denom for SI-based neuron set.")
    ap.add_argument("--eps", type=float, default=1e-8)

    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    _ensure_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # (1) mean(AR) - mean(EN) > thr  (signed means; from mean JSON)
    # -------------------------
    mean_d = json.loads(args.mean_json.read_text(encoding="utf-8"))
    en_mean = np.array(mean_d["en_mean"], dtype=np.float64)  # [L,M]
    ar_mean = np.array(mean_d["ar_mean"], dtype=np.float64)

    delta_signed = ar_mean - en_mean
    thr = float(args.delta_thr)

    mask1 = delta_signed > thr
    vals1 = delta_signed[mask1]
    layers1 = np.nonzero(mask1)[0].tolist()

    L, M = en_mean.shape

    save_hist(
        vals1,
        args.out_dir / f"set1_hist_delta_signed_gt_{thr}.png",
        title=f"Set 1: delta_signed = mean(AR)-mean(EN) > {thr}  (n={vals1.size})",
        xlabel="delta_signed",
        bins=args.bins,
    )
    save_layer_counts(
        per_layer_counts(layers1, n_layers=L),
        args.out_dir / f"set1_layer_counts_delta_signed_gt_{thr}.png",
        title=f"Set 1 per-layer counts (delta_signed > {thr})",
    )

    # -------------------------
    # (2) mean(|act_AR|) - mean(|act_EN|) > thr  (CORRECT abs placement; from .pt activations)
    # -------------------------
    mu_en, mu_ar, used, skipped = compute_abs_means_from_pt(args.act_dir)  # [L,M] each
    delta_absmean = mu_ar - mu_en

    mask2 = delta_absmean > thr
    vals2 = delta_absmean[mask2]
    layers2 = np.nonzero(mask2)[0].tolist()

    save_hist(
        vals2,
        args.out_dir / f"set2_hist_meanabs_diff_gt_{thr}.png",
        title=f"Set 2: mean(|AR|)-mean(|EN|) > {thr}  (n={vals2.size}, used={used}, skipped={skipped})",
        xlabel="mean(|AR|)-mean(|EN|)",
        bins=args.bins,
    )
    save_layer_counts(
        per_layer_counts(layers2, n_layers=L),
        args.out_dir / f"set2_layer_counts_meanabs_diff_gt_{thr}.png",
        title=f"Set 2 per-layer counts (mean(|AR|)-mean(|EN|) > {thr})",
    )

    # -------------------------
    # (3) SI-based (abs-mean over samples):
    # SI = (mu_AR - mu_EN) / (mu_AR + mu_EN + eps)
    # select: SI > si_thr and denom > si_min_denom
    # -------------------------
    denom = mu_ar + mu_en
    si = (mu_ar - mu_en) / (denom + float(args.eps))

    si_thr = float(args.si_thr)
    den_thr = float(args.si_min_denom)

    mask3 = (si > si_thr) & (denom > den_thr)
    vals3 = si[mask3]
    layers3 = np.nonzero(mask3)[0].tolist()

    save_hist(
        vals3,
        args.out_dir / f"set3_hist_si_gt_{si_thr}_den_gt_{den_thr}.png",
        title=f"Set 3: SI>{si_thr} & denom>{den_thr} (n={vals3.size}, used={used}, skipped={skipped})",
        xlabel="SI",
        bins=args.bins,
    )
    save_layer_counts(
        per_layer_counts(layers3, n_layers=L),
        args.out_dir / f"set3_layer_counts_si_gt_{si_thr}_den_gt_{den_thr}.png",
        title=f"Set 3 per-layer counts (SI>{si_thr}, denom>{den_thr})",
    )

    summary = {
        "set1_count": int(vals1.size),
        "set2_count": int(vals2.size),
        "set3_count": int(vals3.size),
        "set2_definition": "mean(|act_AR|) - mean(|act_EN|) > delta_thr",
        "set3_definition": "SI=(mu_AR-mu_EN)/(mu_AR+mu_EN+eps), mu=mean(|act|) over samples",
        "used_activation_files_for_absmean": used,
        "skipped_activation_files_for_absmean": skipped,
        "thresholds": {
            "delta_thr": thr,
            "si_thr": si_thr,
            "si_min_denom": den_thr,
            "eps": float(args.eps),
        },
        "outputs_dir": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote plots to:", args.out_dir)
    print("Wrote summary to:", args.out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

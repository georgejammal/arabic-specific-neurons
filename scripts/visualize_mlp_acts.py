from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_acts(input_dir: Path):
    paths = sorted(input_dir.glob("*.pt"))
    en_list = []
    ar_list = []
    kept_ids = []
    skipped = []

    for p in paths:
        d = torch.load(p, map_location="cpu")
        en = d["en_mlp_gated_lasttok"]
        ar = d["ar_mlp_gated_lasttok"]

        if not torch.isfinite(en).all() or not torch.isfinite(ar).all():
            skipped.append(p.name)
            continue

        en_list.append(en.float())  # [34, 10240]
        ar_list.append(ar.float())  # [34, 10240]
        kept_ids.append(int(p.stem))

    if not en_list:
        raise RuntimeError(f"No valid .pt files found in {input_dir}")

    EN = torch.stack(en_list, dim=0)  # [N, 34, 10240]
    AR = torch.stack(ar_list, dim=0)  # [N, 34, 10240]
    return EN, AR, kept_ids, skipped


def chunk_neurons(x: torch.Tensor, chunk: int) -> torch.Tensor:
    # x: [..., 10240] -> [..., 10240/chunk] by mean pooling
    *prefix, m = x.shape
    m2 = (m // chunk) * chunk
    return x[..., :m2].reshape(*prefix, m2 // chunk, chunk).mean(dim=-1)


def save_heatmap(matrix: torch.Tensor, out_path: Path, title: str, xlabel: str, ylabel: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.imshow(matrix.numpy(), aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_layer_strength_plot(layer_strength: torch.Tensor, out_path: Path, title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 3))
    plt.plot(layer_strength.numpy())
    plt.title(title)
    plt.xlabel("layer (0..33)")
    plt.ylabel("mean |activation|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_hist(values: torch.Tensor, out_path: Path, title: str, xlabel: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    plt.hist(values.numpy(), bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_top_neurons(
    *,
    mean_act: torch.Tensor,  # [34, 10240] signed mean
    layer: int,
    topk: int,
    out_path: Path,
    header: str,
):
    mean_abs = mean_act.abs()
    vals, idx = mean_abs[layer].topk(topk)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        f.write(f"Top {topk} neurons in layer {layer} by |mean activation|\n\n")
        for v, i in zip(vals.tolist(), idx.tolist()):
            f.write(f"neuron={i:5d}  mean={mean_act[layer, i].item(): .6f}  |mean|={v: .6f}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("eval/activations"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval/plots"))
    ap.add_argument("--chunk", type=int, default=64, help="Neuron pooling width for readable heatmaps.")
    ap.add_argument("--layer", type=int, default=33, help="Layer for top-neuron reporting.")
    ap.add_argument("--topk", type=int, default=50, help="Top-K neurons by |mean activation| in --layer.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    EN, AR, kept_ids, skipped = load_acts(args.input_dir)  # [N,34,10240] each
    D = AR - EN  # [N,34,10240] deltas if you still want them

    # -------- English-only stats --------
    en_mean = EN.mean(dim=0)               # [34,10240]
    en_mean_abs = en_mean.abs()            # [34,10240]
    en_layer_strength = en_mean_abs.mean(dim=1)  # [34]
    en_sample_strength = EN.abs().mean(dim=(1, 2))  # [N]

    save_heatmap(
        chunk_neurons(en_mean_abs, args.chunk),
        args.out_dir / "en_mean_abs_heatmap.png",
        title="English: mean |activation| (gated-MLP, last prompt token)",
        xlabel=f"neuron buckets (size={args.chunk})",
        ylabel="layer (0..33)",
    )
    save_layer_strength_plot(
        en_layer_strength,
        args.out_dir / "en_layer_strength.png",
        title="English: layer strength (mean |activation| per layer)",
    )
    save_hist(
        en_sample_strength,
        args.out_dir / "en_sample_strength_hist.png",
        title="English: per-sample mean |activation| distribution",
        xlabel="mean |activation| over (layer, neuron)",
    )
    write_top_neurons(
        mean_act=en_mean,
        layer=int(args.layer),
        topk=int(args.topk),
        out_path=args.out_dir / f"en_top_neurons_layer_{int(args.layer)}.txt",
        header=f"Kept {len(kept_ids)} samples, skipped: {skipped}",
    )

    # -------- Arabic-only stats --------
    ar_mean = AR.mean(dim=0)
    ar_mean_abs = ar_mean.abs()
    ar_layer_strength = ar_mean_abs.mean(dim=1)
    ar_sample_strength = AR.abs().mean(dim=(1, 2))

    save_heatmap(
        chunk_neurons(ar_mean_abs, args.chunk),
        args.out_dir / "ar_mean_abs_heatmap.png",
        title="Arabic: mean |activation| (gated-MLP, last prompt token)",
        xlabel=f"neuron buckets (size={args.chunk})",
        ylabel="layer (0..33)",
    )
    save_layer_strength_plot(
        ar_layer_strength,
        args.out_dir / "ar_layer_strength.png",
        title="Arabic: layer strength (mean |activation| per layer)",
    )
    save_hist(
        ar_sample_strength,
        args.out_dir / "ar_sample_strength_hist.png",
        title="Arabic: per-sample mean |activation| distribution",
        xlabel="mean |activation| over (layer, neuron)",
    )
    write_top_neurons(
        mean_act=ar_mean,
        layer=int(args.layer),
        topk=int(args.topk),
        out_path=args.out_dir / f"ar_top_neurons_layer_{int(args.layer)}.txt",
        header=f"Kept {len(kept_ids)} samples, skipped: {skipped}",
    )

    # -------- (Optional) keep delta heatmap too --------
    mean_delta = D.mean(dim=0)
    save_heatmap(
        chunk_neurons(mean_delta.abs(), args.chunk),
        args.out_dir / "delta_mean_abs_heatmap.png",
        title="Mean |Arabic - English| (gated-MLP, last prompt token)",
        xlabel=f"neuron buckets (size={args.chunk})",
        ylabel="layer (0..33)",
    )

    torch.save(
        {
            "kept_ids": kept_ids,
            "skipped": skipped,
            "en_mean": en_mean,
            "ar_mean": ar_mean,
            "mean_delta": mean_delta,
            "en_layer_strength": en_layer_strength,
            "ar_layer_strength": ar_layer_strength,
            "en_sample_strength": en_sample_strength,
            "ar_sample_strength": ar_sample_strength,
        },
        args.out_dir / "summary_tensors_en_ar.pt",
    )

    print(f"Wrote plots to: {args.out_dir}")
    if skipped:
        print(f"Skipped non-finite files: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

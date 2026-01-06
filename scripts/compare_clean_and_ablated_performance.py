from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Row:
    id: int
    cer: float
    edit_distance: int
    ref_len: int
    exact_clean: bool


@dataclass(frozen=True)
class Report:
    path: Path
    run_name: str
    lang: str
    normalization: str
    mean_cer: float
    rows: dict[int, Row]


def _as_bool(x: Any) -> bool:
    return bool(x)


def load_report(path: Path) -> Report:
    d = json.loads(path.read_text(encoding="utf-8"))

    run_name = str(d.get("run_name") or path.stem)
    lang = str(d.get("lang", ""))
    normalization = str(d.get("normalization", ""))

    rows_list = d.get("rows", [])
    rows: dict[int, Row] = {}
    for r in rows_list:
        i = int(r["id"])
        cer = float(r.get("cer", 0.0))
        dist = int(r.get("edit_distance", 0))
        ref_len = int(r.get("ref_len", 0))

        # Newer reports include exact_clean; older can infer it from normalized strings.
        if "exact_clean" in r:
            exact_clean = _as_bool(r["exact_clean"])
        else:
            exact_clean = str(r.get("gt_norm", "")) == str(r.get("pred_norm", ""))

        rows[i] = Row(id=i, cer=cer, edit_distance=dist, ref_len=ref_len, exact_clean=exact_clean)

    mean_cer = float(d.get("mean_cer", 0.0))
    return Report(
        path=path,
        run_name=run_name,
        lang=lang,
        normalization=normalization,
        mean_cer=mean_cer,
        rows=rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare clean CER reports (baseline vs ablated/patched).")
    ap.add_argument("--base", type=Path, required=True, help="Baseline CER JSON (e.g., ar_clean_...__gemma_ocr_results.json)")
    ap.add_argument(
        "--ablated",
        type=Path,
        required=True,
        help="Ablated/patched CER JSON (e.g., ar_clean_...__gemma_ocr_results_patch_ar_to_en_acts.json)",
    )
    ap.add_argument("--topk", type=int, default=15, help="Show top-K improvements/regressions.")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output with per-id deltas.")
    args = ap.parse_args()

    base = load_report(args.base)
    ab = load_report(args.ablated)

    if base.lang and ab.lang and base.lang != ab.lang:
        print(f"WARNING: lang differs: base={base.lang} ablated={ab.lang}")
    if base.normalization and ab.normalization and base.normalization != ab.normalization:
        print("WARNING: normalization differs:")
        print("  base   :", base.normalization)
        print("  ablated:", ab.normalization)

    ids = sorted(set(base.rows) & set(ab.rows))
    if not ids:
        raise SystemExit("No overlapping ids found between the two reports.")

    improved = worse = same = 0
    exact_gain = exact_loss = 0
    dist_base_sum = dist_ab_sum = 0

    deltas: list[tuple[float, int]] = []
    per_id: list[dict[str, Any]] = []

    for i in ids:
        b = base.rows[i]
        a = ab.rows[i]
        dcer = b.cer - a.cer

        dist_base_sum += b.edit_distance
        dist_ab_sum += a.edit_distance

        if dcer > 0:
            improved += 1
        elif dcer < 0:
            worse += 1
        else:
            same += 1

        if (not b.exact_clean) and a.exact_clean:
            exact_gain += 1
        if b.exact_clean and (not a.exact_clean):
            exact_loss += 1

        deltas.append((dcer, i))
        per_id.append(
            {
                "id": i,
                "cer_base": b.cer,
                "cer_ablated": a.cer,
                "delta_cer_base_minus_ablated": dcer,
                "edit_distance_base": b.edit_distance,
                "edit_distance_ablated": a.edit_distance,
                "exact_clean_base": b.exact_clean,
                "exact_clean_ablated": a.exact_clean,
            }
        )

    mean_base = sum(base.rows[i].cer for i in ids) / len(ids)
    mean_ab = sum(ab.rows[i].cer for i in ids) / len(ids)

    print("BASE:", base.path)
    print("ABL :", ab.path)
    print(f"n={len(ids)}")
    print(f"mean_CER(clean): base={mean_base:.6f}  ablated={mean_ab:.6f}  Δ(base-ablated)={(mean_base-mean_ab):.6f}")
    print(f"sum_edit_distance: base={dist_base_sum}  ablated={dist_ab_sum}  edits_saved={dist_base_sum-dist_ab_sum}")
    print(f"exact_clean: base={sum(base.rows[i].exact_clean for i in ids)}/{len(ids)}  ablated={sum(ab.rows[i].exact_clean for i in ids)}/{len(ids)}")
    print(f"exact_clean gains={exact_gain} losses={exact_loss}")
    print(f"per-id: improved={improved} worse={worse} same={same}")

    deltas.sort(reverse=True)  # biggest improvement first
    k = int(args.topk)

    print(f"\nTop {k} improvements (cer_base - cer_ablated):")
    for dcer, i in deltas[:k]:
        b = base.rows[i]
        a = ab.rows[i]
        print(f"id={i:03d}  ΔCER={dcer:.4f}  dist {b.edit_distance}->{a.edit_distance}  CER {b.cer:.4f}->{a.cer:.4f}")

    print(f"\nTop {k} regressions:")
    for dcer, i in deltas[-k:][::-1]:
        b = base.rows[i]
        a = ab.rows[i]
        print(f"id={i:03d}  ΔCER={dcer:.4f}  dist {b.edit_distance}->{a.edit_distance}  CER {b.cer:.4f}->{a.cer:.4f}")

    if args.out:
        payload = {
            "base": str(base.path),
            "ablated": str(ab.path),
            "n": len(ids),
            "mean_cer_base": mean_base,
            "mean_cer_ablated": mean_ab,
            "mean_cer_delta_base_minus_ablated": (mean_base - mean_ab),
            "sum_edit_distance_base": dist_base_sum,
            "sum_edit_distance_ablated": dist_ab_sum,
            "sum_edit_distance_saved": (dist_base_sum - dist_ab_sum),
            "improved": improved,
            "worse": worse,
            "same": same,
            "exact_gain": exact_gain,
            "exact_loss": exact_loss,
            "per_id": sorted(per_id, key=lambda r: r["id"]),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nWrote:", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

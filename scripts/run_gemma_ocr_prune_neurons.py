from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
PAIRS_JSON = ROOT / "datasets" / "text_dataset.json"
IMG_ROOT = ROOT / "image_dataset"


def require_cuda_or_exit() -> None:
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA GPU required.")


def require_bf16_or_exit() -> None:
    require_cuda_or_exit()
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported) and not is_bf16_supported():
        sys.exit("ERROR: This GPU/driver does not support CUDA bf16; fp16 support is intentionally disabled.")

    # Fallback probe for older torch versions.
    try:
        x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
        _ = x @ x
    except Exception as e:
        sys.exit(f"ERROR: CUDA bf16 not usable on this system; fp16 support is intentionally disabled. ({e})")


def parse_dtype(name: Literal["bf16"]) -> torch.dtype:
    if name != "bf16":
        raise ValueError(name)
    return torch.bfloat16


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processor_and_model(
    *,
    model_id: str,
    dtype: torch.dtype,
    quant: Literal["none", "8bit"],
) -> tuple[AutoProcessor, Gemma3ForConditionalGeneration]:
    require_bf16_or_exit()

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    kwargs: dict[str, object] = {
        "device_map": "cuda",
        "torch_dtype": dtype,
    }

    if quant == "8bit":
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception as e:
            sys.exit(f"ERROR: 8-bit requested but BitsAndBytesConfig unavailable: {e}")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=dtype)

    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
    except Exception as e:
        sys.exit(f"ERROR: Failed to load model (quant={quant}, dtype={dtype}): {e}")

    if quant == "none":
        try:
            model = model.to(dtype=dtype)
        except Exception as e:
            sys.exit(f"ERROR: Failed to cast model to bf16 on CUDA (fp16 disabled): {e}")

    return processor, model


def build_chat_prompt(processor: AutoProcessor, instruction: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def load_prune_set(path: Path) -> dict[int, list[int]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("rows", d)
    by_layer: dict[int, list[int]] = {}
    for r in rows:
        layer = int(r["layer"])
        neuron = int(r["neuron"])
        by_layer.setdefault(layer, []).append(neuron)
    for k in list(by_layer.keys()):
        by_layer[k] = sorted(set(by_layer[k]))
    if not by_layer:
        sys.exit(f"No neurons found in {path}")
    return by_layer


def _parse_layer_idx_from_name(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except Exception:
                continue
    return None


def sample_random_neurons(
    model: Gemma3ForConditionalGeneration,
    *,
    n: int,
    seed: int | None = None,
) -> dict[int, list[int]]:
    """
    Sample `n` random (layer, neuron) positions across all Gemma3MLP layers.
    """
    if n <= 0:
        sys.exit("--random-n must be > 0")

    rng = random.Random(seed)
    candidates: list[tuple[int, int]] = []

    for name, module in model.named_modules():
        if module.__class__.__name__ != "Gemma3MLP":
            continue
        layer_idx = _parse_layer_idx_from_name(name)
        if layer_idx is None:
            continue
        hidden = module.gate_proj.weight.shape[0]
        candidates.extend((layer_idx, i) for i in range(hidden))

    if not candidates:
        sys.exit("No Gemma3MLP layers found when sampling random neurons.")
    if n > len(candidates):
        sys.exit(f"--random-n={n} exceeds total neurons {len(candidates)}")

    chosen = rng.sample(candidates, n)

    by_layer: dict[int, list[int]] = {}
    for layer_idx, neuron_idx in chosen:
        by_layer.setdefault(layer_idx, []).append(neuron_idx)
    for k in by_layer:
        by_layer[k] = sorted(set(by_layer[k]))
    return by_layer


def build_prune_by_layer(
    model: Gemma3ForConditionalGeneration,
    *,
    prune_set_path: Path | None,
    random_n: int,
    random_seed: int | None,
) -> dict[int, list[int]]:
    if random_n > 0:
        print(f"Sampling {random_n} random neurons (seed={random_seed})")
        return sample_random_neurons(model, n=random_n, seed=random_seed)
    if prune_set_path is not None:
        return load_prune_set(prune_set_path)
    sys.exit("Must provide either --prune-set or --random-n > 0.")


def register_prune_hooks(model: Gemma3ForConditionalGeneration, *, prune_by_layer: dict[int, list[int]]):
    """
    Apply an operation on selected gated-intermediate neurons for all tokens (prefill + decode).
    """
    handles = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Gemma3MLP":
            continue

        layer_idx = _parse_layer_idx_from_name(name)
        if layer_idx is None or layer_idx not in prune_by_layer:
            continue

        idx = torch.tensor(prune_by_layer[layer_idx], dtype=torch.long, device=module.down_proj.weight.device)

        # Precompute a per-layer multiplier vector to avoid advanced indexing each call.
        mult = torch.ones(module.gate_proj.weight.shape[0], device=module.down_proj.weight.device, dtype=torch.float32)
        mult[idx] = 0.0
        mult_minus_one = mult - 1.0

        def hook(mod, inputs, output, *, multm1_local=mult_minus_one):
            # inputs[0]: [B, T, hidden_size]; output: [B, T, hidden_size]
            x = inputs[0]
            if x.ndim != 3:
                return output

            gate = mod.gate_proj(x).float()  # [B,T,M]
            up = mod.up_proj(x).float()
            inter = mod.act_fn(gate) * up    # [B,T,M], float32

            delta = inter * multm1_local     # broadcast over [B,T,M]

            delta_out = mod.down_proj(delta.to(dtype=mod.down_proj.weight.dtype))
            delta_out = delta_out.to(dtype=output.dtype)

            return output + delta_out

        handles.append(module.register_forward_hook(hook))

    if not handles:
        sys.exit("ERROR: No prune hooks registered (layer parsing mismatch).")
    return handles


def register_op_hooks(
    model: Gemma3ForConditionalGeneration,
    *,
    prune_by_layer: dict[int, list[int]],
    op: str,
    scale_factor: float,
):
    """
    Apply either:
      - op='prune': set selected gated-intermediate dims to 0
      - op='scale': multiply selected dims by scale_factor
    for all tokens (prefill + decode).
    """
    if op not in {"prune", "scale"}:
        sys.exit(f"Unknown --op {op!r}")
    if op == "scale" and scale_factor <= 0:
        sys.exit("--scale-factor must be > 0")

    handles = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Gemma3MLP":
            continue

        layer_idx = _parse_layer_idx_from_name(name)
        if layer_idx is None or layer_idx not in prune_by_layer:
            continue

        idx = torch.tensor(prune_by_layer[layer_idx], dtype=torch.long, device=module.down_proj.weight.device)

        mult = torch.ones(module.gate_proj.weight.shape[0], device=module.down_proj.weight.device, dtype=torch.float32)
        if op == "prune":
            mult[idx] = 0.0
        else:
            mult[idx] = float(scale_factor)
        mult_minus_one = mult - 1.0

        def hook(mod, inputs, output, *, multm1_local=mult_minus_one):
            x = inputs[0]
            if x.ndim != 3:
                return output

            gate = mod.gate_proj(x).float()  # [B,T,M]
            up = mod.up_proj(x).float()
            inter = mod.act_fn(gate) * up    # [B,T,M], float32

            delta = inter * multm1_local
            delta_out = mod.down_proj(delta.to(dtype=mod.down_proj.weight.dtype))
            delta_out = delta_out.to(dtype=output.dtype)
            return output + delta_out

        handles.append(module.register_forward_hook(hook))

    if not handles:
        sys.exit("ERROR: No op hooks registered (layer parsing mismatch).")
    return handles


@torch.inference_mode()
def generate_ocr(
    *,
    processor: AutoProcessor,
    model: Gemma3ForConditionalGeneration,
    prompt_text: str,
    image_path: Path,
    max_new_tokens: int,
) -> str:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt_text, images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = out[0, input_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Gemma OCR with neuron pruning (prefill + decode, all tokens).")
    ap.add_argument("--model-id", type=str, default="google/gemma-3-4b-it")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--instruction", type=str, default="Read the text in the image. Return only the text, with no explanation.")
    ap.add_argument("--data-json", type=Path, default=PAIRS_JSON, help="Path to text dataset JSON (default: datasets/text_dataset.json).")
    ap.add_argument("--img-root", type=Path, default=IMG_ROOT, help="Image root containing en/ and ar/ subfolders (default: image_dataset).")
    ap.add_argument("--langs", choices=["both", "ar", "en"], default="both", help="Which language(s) to run OCR for.")
    ap.add_argument("--quant", choices=["none", "8bit"], default="none", help="Optional 8-bit load to reduce VRAM (requires bitsandbytes).")
    ap.add_argument("--dtype", choices=["bf16"], default="bf16", help="Model dtype (bf16 only; fp16 intentionally disabled).")
    ap.add_argument("--op", choices=["prune", "scale"], default="prune", help="Operation on selected neurons (default: prune).")
    ap.add_argument("--scale-factor", type=float, default=1.5, help="If --op=scale, multiply selected neurons by this factor.")
    ap.add_argument("--prune-set", type=Path, required=False, default=None, help="Path to neuron set JSON (e.g., setA_delta_gt_1.2.json). Ignored if --random-n > 0.")
    ap.add_argument("--random-n", type=int, default=0, help="If >0, sample this many random neurons to prune instead of using --prune-set.")
    ap.add_argument("--random-seed", type=int, default=None, help="Optional seed for --random-n sampling.")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "gemma_ocr_results_prune.jsonl")
    ap.add_argument("--print-every", type=int, default=1)
    args = ap.parse_args()

    require_bf16_or_exit()

    seed_everything(args.seed)

    pairs = json.loads(args.data_json.read_text(encoding="utf-8"))
    n = min(int(args.n), len(pairs))

    processor, model = load_processor_and_model(
        model_id=args.model_id,
        dtype=parse_dtype(str(args.dtype)),
        quant=str(args.quant),  # type: ignore[arg-type]
    )

    prune_by_layer = build_prune_by_layer(
        model,
        prune_set_path=args.prune_set,
        random_n=args.random_n,
        random_seed=args.random_seed,
    )
    handles = register_op_hooks(
        model,
        prune_by_layer=prune_by_layer,
        op=str(args.op),
        scale_factor=float(args.scale_factor),
    )
    prompt_text = build_chat_prompt(processor, args.instruction)

    prune_set_label = (
        f"random_n={args.random_n}_seed={args.random_seed}"
        if args.random_n > 0
        else str(args.prune_set)
    )
    op_label = (
        "prune"
        if args.op == "prune"
        else f"scale{args.scale_factor:g}"
    )

    do_en = args.langs in ("both", "en")
    do_ar = args.langs in ("both", "ar")
    if not (do_en or do_ar):
        sys.exit("ERROR: --langs must include at least one of 'en' or 'ar'.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(n):
            en_img = args.img_root / "en" / f"{i:03d}.png"
            ar_img = args.img_root / "ar" / f"{i:03d}.png"

            en_gt = str(pairs[i].get("en", ""))
            ar_gt = str(pairs[i].get("ar", ""))

            if do_en:
                en_pred = generate_ocr(
                    processor=processor,
                    model=model,
                    prompt_text=prompt_text,
                    image_path=en_img,
                    max_new_tokens=args.max_new_tokens,
                )
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "lang": "en",
                            "image_path": str(en_img),
                            "gt_text": en_gt,
                            "pred_text": en_pred,
                            "model_id": args.model_id,
                            "prune_set": prune_set_label,
                            "op": op_label,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if do_ar:
                ar_pred = generate_ocr(
                    processor=processor,
                    model=model,
                    prompt_text=prompt_text,
                    image_path=ar_img,
                    max_new_tokens=args.max_new_tokens,
                )
                f.write(
                    json.dumps(
                        {
                            "id": i,
                            "lang": "ar",
                            "image_path": str(ar_img),
                            "gt_text": ar_gt,
                            "pred_text": ar_pred,
                            "model_id": args.model_id,
                            "prune_set": prune_set_label,
                            "op": op_label,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if args.print_every > 0 and (i % args.print_every == 0 or i == n - 1):
                print(f"[{i+1:03d}/{n:03d}] wrote to {args.out}")

    for h in handles:
        h.remove()

    print("Done:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

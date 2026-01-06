from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = ROOT / "datasets" / "text_dataset.json"
IMG_ROOT = ROOT / "image_dataset"
OUT_DIR_DEFAULT = ROOT / "eval" / "activations"

MODEL_ID_DEFAULT = "google/gemma-3-4b-it"


def require_cuda_or_exit() -> None:
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA GPU required (no CPU fallback).")


def build_chat_prompt(processor: AutoProcessor, instruction: str) -> str:
    # Gemma3 multimodal needs an explicit image placeholder token.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def load_model(model_id: str, dtype_name: str) -> tuple[AutoProcessor, Gemma3ForConditionalGeneration, torch.dtype]:
    require_cuda_or_exit()

    if dtype_name == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype {dtype_name!r}")

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=dtype,
    ).eval()
    return processor, model, dtype


def register_mlp_hooks(model: Gemma3ForConditionalGeneration):
    """
    Registers hooks on every Gemma3MLP. On each forward pass, we compute the
    gated MLP intermediate at *one token position*:
        act_fn(gate_proj(x_tok)) * up_proj(x_tok)
    and stash it in a per sample buffer.

    We do this in the hook because the module forward returns down_proj(...),
    but we want the non-linear neuron activations inside the MLP.
    """
    mlp_modules: list[tuple[int, object]] = []

    for name, module in model.named_modules():
        if module.__class__.__name__ == "Gemma3MLP":
            # name looks like: model.language_model.layers.{L}.mlp
            # Extract layer index defensively.
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except Exception:
                        pass
            if layer_idx is None:
                continue
            mlp_modules.append((layer_idx, module))

    if not mlp_modules:
        sys.exit("ERROR: Could not find Gemma3MLP modules to hook.")

    mlp_modules.sort(key=lambda x: x[0])
    n_layers = len(mlp_modules)

    # Per-forward context (mutated before each model call)
    ctx = {
        "last_idx": None,  # int
        "acts_by_layer": None,  # list[torch.Tensor]
    }

    handles = []

    def make_hook(layer_index: int):
        def hook(module, inputs, output):
            # inputs[0] is the hidden state x: [B, T, hidden_size]
            x = inputs[0]
            last_idx = ctx["last_idx"]
            if last_idx is None:
                raise RuntimeError("Hook context last_idx is not set.")

            # Take only the last prompt token state: [B, hidden_size]
            x_tok = x[:, last_idx, :]

            # Compute gated intermediate: act_fn(gate_proj(x)) * up_proj(x)
            gate = module.gate_proj(x_tok)
            up = module.up_proj(x_tok)
            act = module.act_fn(gate) * up  # [B, intermediate_size]

            # Store CPU tensor for this layer (batch assumed 1)
            ctx["acts_by_layer"][layer_index] = act[0].detach().to("cpu")

        return hook

    for i, (layer_idx, module) in enumerate(mlp_modules):
        # We store sequentially [0..n_layers-1] in increasing layer_idx order.
        handles.append(module.register_forward_hook(make_hook(i)))

    return ctx, handles, n_layers


@torch.inference_mode()
def extract_one(
    *,
    processor: AutoProcessor,
    model: Gemma3ForConditionalGeneration,
    prompt_text: str,
    image_path: Path,
    ctx: dict,
    n_layers: int,
) -> torch.Tensor:
    """
    Runs a single forward pass (no generation) and returns:
      activations: [n_layers, intermediate_size] (CPU)
    corresponding to the last prompt token.
    """
    img = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt_text, images=img, return_tensors="pt")

    # We stop before generation, so the "last prompt token" is simply the last input id.
    last_idx = inputs["input_ids"].shape[-1] - 1

    # Prepare hook buffers
    ctx["last_idx"] = int(last_idx)
    ctx["acts_by_layer"] = [None] * n_layers

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward only (no model.generate)
    _ = model(**inputs, use_cache=False)

    acts = ctx["acts_by_layer"]
    if any(a is None for a in acts):
        raise RuntimeError("Some layer activations were not captured (hook mismatch).")

    return torch.stack(acts, dim=0)  # [n_layers, intermediate_size] on CPU


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract last-prompt-token MLP activations for Gemma3 OCR inputs.")
    parser.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument(
        "--data-json",
        type=Path,
        default=DATA_JSON,
        help="JSON file with a list of {en, ar} pairs (default: datasets/text_dataset.json).",
    )
    parser.add_argument(
        "--img-root",
        type=Path,
        default=IMG_ROOT,
        help="Image root containing en/ and ar/ subfolders (default: image_dataset).",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Read the text in the image. Return only the text, with no explanation.",
        help="Use the SAME instruction for EN and AR to avoid prompt confounds.",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of ids to process (default: 100).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--print-every", type=int, default=1)
    args = parser.parse_args()

    pairs = json.loads(args.data_json.read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        sys.exit(f"ERROR: Expected a list in {args.data_json}")

    processor, model, dtype = load_model(args.model_id, args.dtype)
    print(f"Using dtype: {dtype}")
    prompt_text = build_chat_prompt(processor, args.instruction)

    ctx, handles, n_layers = register_mlp_hooks(model)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(int(args.n), len(pairs))
    for i in range(n):
        en_img = args.img_root / "en" / f"{i:03d}.png"
        ar_img = args.img_root / "ar" / f"{i:03d}.png"

        en_acts = extract_one(
            processor=processor,
            model=model,
            prompt_text=prompt_text,
            image_path=en_img,
            ctx=ctx,
            n_layers=n_layers,
        )
        ar_acts = extract_one(
            processor=processor,
            model=model,
            prompt_text=prompt_text,
            image_path=ar_img,
            ctx=ctx,
            n_layers=n_layers,
        )

        # Save one file per id with both languages (easy pairing for deltas later).
        payload = {
            "id": i,
            "model_id": args.model_id,
            "dtype": args.dtype,
            "instruction": args.instruction,
            "en_image_path": str(en_img),
            "ar_image_path": str(ar_img),
            # tensors: [34, 10240] each (CPU). Use bf16 storage to reduce size.
            "en_mlp_gated_lasttok": en_acts.to(dtype=torch.bfloat16),
            "ar_mlp_gated_lasttok": ar_acts.to(dtype=torch.bfloat16),
        }
        torch.save(payload, out_dir / f"{i:03d}.pt")

        if args.print_every > 0 and (i % args.print_every == 0 or i == n - 1):
            print(f"[{i+1:03d}/{n:03d}] saved {out_dir / f'{i:03d}.pt'}")

    for h in handles:
        h.remove()

    print(f"Done. Saved {n} paired activation files to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

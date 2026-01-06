from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


# -----------------------------
# Project paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = ROOT / "datasets" / "text_dataset.json"
IMG_ROOT = ROOT / "image_dataset"
EVAL_DIR = ROOT / "eval"

MODEL_ID_DEFAULT = "google/gemma-3-4b-it"  # multimodal (image+text)


def normalize_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def require_cuda_or_exit() -> None:
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA GPU required (no CPU fallback).")


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


def build_instruction(lang: str) -> str:
    if lang == "ar":
        return "اقرأ النص في الصورة. أعد النص فقط كما هو، بدون شرح."
    return "Read the text in the image. Return only the text, with no explanation."


def build_chat_prompt(processor: AutoProcessor, instruction: str) -> str:
    # Gemma3 multimodal requires an explicit image placeholder token in the prompt.
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


@torch.inference_mode()
def ocr_one(
    *,
    processor: AutoProcessor,
    model: Gemma3ForConditionalGeneration,
    image_path: Path,
    lang: str,
    max_new_tokens: int,
) -> str:
    img = Image.open(image_path).convert("RGB")
    prompt_text = build_chat_prompt(processor, build_instruction(lang))

    inputs = processor(text=prompt_text, images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the newly generated tokens (avoid printing/recording the prompt).
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = out[0, input_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemma OCR on the generated English/Arabic image dataset.")
    parser.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--n", type=int, default=100, help="How many ids to run (default: all).")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--data-json", type=Path, default=DATA_JSON, help="Path to text dataset JSON.")
    parser.add_argument("--img-root", type=Path, default=IMG_ROOT, help="Image root containing en/ and ar/ subfolders.")
    parser.add_argument("--langs", choices=["both", "ar", "en"], default="both", help="Which language(s) to run OCR for.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")

    # Quantization control (try 8-bit if you want; exits if it fails).
    parser.add_argument("--quant", choices=["none", "8bit"], default="none")

    # Dtype control: bf16 only (fp16 intentionally disabled).
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")

    # Monitoring/progress
    parser.add_argument("--print-every", type=int, default=1, help="Print a progress line every N ids.")
    parser.add_argument("--show-preds", action="store_true", help="Also print GT/PRED text as it runs.")

    # Output (JSONL written incrementally so you can monitor it live)
    parser.add_argument(
        "--out",
        type=Path,
        default=EVAL_DIR / "gemma_ocr_results.jsonl",
        help="JSONL output path (written incrementally).",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    pairs = json.loads(args.data_json.read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        sys.exit(f"ERROR: Expected a list in {args.data_json}")

    n = min(int(args.n), len(pairs))

    # Load the model ONCE, then reuse for all images.
    processor, model = load_processor_and_model(
        model_id=args.model_id,
        dtype=parse_dtype(args.dtype),
        quant=args.quant,  # "none" or "8bit"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with args.out.open("w", encoding="utf-8") as f:
        do_en = args.langs in ("both", "en")
        do_ar = args.langs in ("both", "ar")
        if not (do_en or do_ar):
            sys.exit("ERROR: --langs must include at least one of 'en' or 'ar'.")

        for i in range(n):
            en_img = args.img_root / "en" / f"{i:03d}.png"
            ar_img = args.img_root / "ar" / f"{i:03d}.png"

            en_gt = str(pairs[i].get("en", ""))
            ar_gt = str(pairs[i].get("ar", ""))

            wrote = 0

            if do_en:
                en_pred = ocr_one(
                    processor=processor,
                    model=model,
                    image_path=en_img,
                    lang="en",
                    max_new_tokens=args.max_new_tokens,
                )
                en_rec = {
                    "id": i,
                    "lang": "en",
                    "model_id": args.model_id,
                    "quant": args.quant,
                    "dtype": args.dtype,
                    "image_path": str(en_img),
                    "gt_text": en_gt,
                    "pred_text": en_pred,
                    "exact_norm": normalize_ws(en_gt) == normalize_ws(en_pred),
                }
                f.write(json.dumps(en_rec, ensure_ascii=False) + "\n")
                wrote += 1

            if do_ar:
                ar_pred = ocr_one(
                    processor=processor,
                    model=model,
                    image_path=ar_img,
                    lang="ar",
                    max_new_tokens=args.max_new_tokens,
                )
                ar_rec = {
                    "id": i,
                    "lang": "ar",
                    "model_id": args.model_id,
                    "quant": args.quant,
                    "dtype": args.dtype,
                    "image_path": str(ar_img),
                    "gt_text": ar_gt,
                    "pred_text": ar_pred,
                    "exact_norm": normalize_ws(ar_gt) == normalize_ws(ar_pred),
                }
                f.write(json.dumps(ar_rec, ensure_ascii=False) + "\n")
                wrote += 1

            f.flush()

            # Terminal monitoring
            if args.print_every > 0 and (i % args.print_every == 0 or i == n - 1):
                elapsed = time.time() - started
                langs_label = "EN+AR" if wrote == 2 else "EN" if do_en else "AR"
                print(f"[{i+1:03d}/{n:03d}] wrote {langs_label} -> {args.out} (elapsed {elapsed:.1f}s)")

            if args.show_preds:
                print("ID", i)
                if do_en:
                    print("GT EN:", en_gt)
                    print("PR EN:", en_pred)
                if do_ar:
                    print("GT AR:", ar_gt)
                    print("PR AR:", ar_pred)
                print()

    rows = n * (2 if args.langs == "both" else 1)
    print(f"Done. Wrote {rows} JSONL rows to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_JSON_PATH = PROJECT_ROOT / "datasets" / "text_dataset.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "image_dataset"


@dataclass(frozen=True)
class FontChoice:
    path: str | None
    size: int


def _candidate_font_paths() -> list[str]:
    roots = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        str(Path.home() / ".fonts"),
        str(Path.home() / ".local/share/fonts"),
    ]
    exts = {".ttf", ".otf", ".ttc"}
    patterns = [
        # Arabic-preferred
        "NotoNaskhArabic",
        "NotoSansArabic",
        "Amiri",
        "Scheherazade",
        "KacstNaskh",
        # Hebrew-preferred
        "NotoSansHebrew",
        "NotoSerifHebrew",
        "Ezra",
        "FrankRuehl",
        # CJK-preferred
        "NotoSansSC",
        "NotoSansTC",
        "NotoSansCJK",
        "SourceHanSans",
        "WenQuanYi",
        "SimHei",
        # Latin fallback
        "DejaVuSans",
        "LiberationSans",
        "Arial",
    ]

    found: list[str] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for fp in p.rglob("*"):
            if fp.suffix.lower() not in exts:
                continue
            if any(tok in fp.name for tok in patterns):
                found.append(str(fp))

    # stable dedup
    seen: set[str] = set()
    out: list[str] = []
    for x in found:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def choose_font_for_lang(lang: str, size: int, explicit_path: str | None) -> FontChoice:
    if explicit_path:
        return FontChoice(path=explicit_path, size=size)

    candidates = _candidate_font_paths()

    def pick(tokens: Iterable[str]) -> str | None:
        for fp in candidates:
            if any(t in Path(fp).name for t in tokens):
                return fp
        return None

    if lang == "ar":
        path = pick(["NotoNaskhArabic", "NotoSansArabic", "Amiri", "Scheherazade", "KacstNaskh", "DejaVuSans"])
    elif lang == "he":
        path = pick(["NotoSansHebrew", "NotoSerifHebrew", "Ezra", "FrankRuehl", "DejaVuSans"])
    elif lang == "zh":
        path = pick(["NotoSansSC", "NotoSansTC", "NotoSansCJK", "SourceHanSans", "WenQuanYi", "SimHei", "DejaVuSans"])
    else:
        path = pick(["DejaVuSans", "LiberationSans", "Arial"])

    return FontChoice(path=path, size=size)


def load_font(choice: FontChoice) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if choice.path:
        try:
            return ImageFont.truetype(choice.path, choice.size)
        except Exception:
            pass
    return ImageFont.load_default()


def _maybe_shape_arabic(text: str) -> str:
    # Optional: best Arabic appearance requires these deps.
    try:
        import arabic_reshaper  # type: ignore
        from bidi.algorithm import get_display  # type: ignore

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _display_text(text: str, *, lang: str) -> str:
    return _maybe_shape_arabic(text) if lang == "ar" else text


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int, *, lang: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return [""]

    tokens = text.split()
    sep = " "

    def width_of(s: str) -> int:
        s = _display_text(s, lang=lang)
        bbox = draw.textbbox((0, 0), s, font=font)
        return int(bbox[2] - bbox[0])

    lines: list[str] = []
    cur: list[str] = []

    for tok in tokens:
        trial = (sep.join(cur + [tok])).strip() if cur else tok
        if width_of(trial) <= max_width:
            cur.append(tok)
            continue

        if cur:
            lines.append(sep.join(cur))
            cur = [tok]
            continue

        # Single token too long: hard-split it.
        chunk = tok
        while chunk:
            lo, hi = 1, len(chunk)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                part = chunk[:mid]
                if width_of(part) <= max_width:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            lines.append(chunk[:best])
            chunk = chunk[best:]

    if cur:
        lines.append(sep.join(cur))

    # shape per-line so width/bbox matches what we draw
    return [_display_text(line, lang=lang) for line in lines] if lang == "ar" else lines


def fit_text(
    *,
    img_size: int,
    padding_x: int,
    padding_y: int,
    text: str,
    lang: str,
    font_path: str | None,
    start_font_size: int,
    min_font_size: int,
    line_spacing: int,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], tuple[int, int]]:
    tmp = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(tmp)

    max_w = max(1, img_size - 2 * padding_x)
    max_h = max(1, img_size - 2 * padding_y)

    align = "right" if lang == "ar" else "left"

    for size in range(start_font_size, min_font_size - 1, -2):
        font = load_font(choose_font_for_lang(lang, size=size, explicit_path=font_path))
        lines = wrap_text(draw, text, font, max_w, lang=lang)
        bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=line_spacing, align=align)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
        if text_w <= max_w and text_h <= max_h:
            return font, lines, (text_w, text_h)

    font = load_font(choose_font_for_lang(lang, size=min_font_size, explicit_path=font_path))
    lines = wrap_text(draw, text, font, max_w, lang=lang)
    bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=line_spacing, align=align)
    text_w = int(bbox[2] - bbox[0])
    text_h = int(bbox[3] - bbox[1])
    return font, lines, (text_w, text_h)


def render_square(
    *,
    text: str,
    lang: str,
    out_path: Path,
    img_size: int,
    padding_x: int,
    padding_y: int,
    font_path: str | None,
    start_font_size: int,
    min_font_size: int,
    line_spacing: int,
) -> None:
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    font, lines, (text_w, text_h) = fit_text(
        img_size=img_size,
        padding_x=padding_x,
        padding_y=padding_y,
        text=text,
        lang=lang,
        font_path=font_path,
        start_font_size=start_font_size,
        min_font_size=min_font_size,
        line_spacing=line_spacing,
    )

    if lang == "ar":
        x = img_size - padding_x - text_w
        align = "right"
    else:
        x = padding_x
        align = "left"

    # vertically center but respect padding
    y = max(padding_y, int((img_size - text_h) / 2))
    y = min(y, img_size - padding_y - text_h)

    draw.multiline_text(
        (x, y),
        "\n".join(lines),
        fill="black",
        font=font,
        spacing=line_spacing,
        align=align,
    )
    img.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate square text images for English/Arabic pairs.")
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--img-size", type=int, default=896)
    parser.add_argument("--padding-x", type=int, default=80)
    parser.add_argument("--padding-y", type=int, default=120)
    parser.add_argument("--font-size", type=int, default=56, help="Starting font size (auto-shrinks to fit).")
    parser.add_argument("--min-font-size", type=int, default=28)
    parser.add_argument("--line-spacing", type=int, default=10)
    parser.add_argument("--font-path-en", type=str, default=None)
    parser.add_argument("--font-path-ar", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only render the first N pairs.")
    parser.add_argument("--lang-only", choices=["en", "ar", "he", "zh"], default=None, help="Render only one language.")
    parser.add_argument("--text-key", type=str, default="text", help="Key to use for lang-only datasets.")
    args = parser.parse_args()

    pairs = json.loads(args.json_path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        raise ValueError(f"Expected a list in {args.json_path}")

    output_root: Path = args.output_root
    if args.lang_only:
        lang_dir = output_root / args.lang_only
        lang_dir.mkdir(parents=True, exist_ok=True)
    else:
        en_dir = output_root / "en"
        ar_dir = output_root / "ar"
        en_dir.mkdir(parents=True, exist_ok=True)
        ar_dir.mkdir(parents=True, exist_ok=True)

    n = len(pairs) if args.limit is None else min(len(pairs), int(args.limit))
    for idx in range(n):
        pair = pairs[idx]
        fname = f"{idx:03d}.png"

        if args.lang_only:
            if isinstance(pair, dict):
                text = pair.get(args.text_key, pair.get(args.lang_only, ""))
            else:
                text = pair
            render_square(
                text=str(text),
                lang=args.lang_only,
                out_path=lang_dir / fname,
                img_size=int(args.img_size),
                padding_x=int(args.padding_x),
                padding_y=int(args.padding_y),
                font_path=args.font_path_ar if args.lang_only == "ar" else args.font_path_en,
                start_font_size=int(args.font_size),
                min_font_size=int(args.min_font_size),
                line_spacing=int(args.line_spacing),
            )
        else:
            render_square(
                text=str(pair.get("en", "")),
                lang="en",
                out_path=en_dir / fname,
                img_size=int(args.img_size),
                padding_x=int(args.padding_x),
                padding_y=int(args.padding_y),
                font_path=args.font_path_en,
                start_font_size=int(args.font_size),
                min_font_size=int(args.min_font_size),
                line_spacing=int(args.line_spacing),
            )

            render_square(
                text=str(pair.get("ar", "")),
                lang="ar",
                out_path=ar_dir / fname,
                img_size=int(args.img_size),
                padding_x=int(args.padding_x),
                padding_y=int(args.padding_y),
                font_path=args.font_path_ar,
                start_font_size=int(args.font_size),
                min_font_size=int(args.min_font_size),
                line_spacing=int(args.line_spacing),
            )

    print(f"Image dataset created at: {output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

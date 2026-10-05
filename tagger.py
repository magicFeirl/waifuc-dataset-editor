"""
Unified tagger interface.

Change BACKEND to switch between backends:
    "pixai"  —  PixAI Tagger v1.0  (default)
    "wd"     —  WD-EVA02-Large v3
"""

import os
from pathlib import Path

# os.environ["HF_HUB_CACHE"] = str(Path(__file__).parent / "model_cache")

# ============================================================
# Backend selection
# ============================================================
BACKEND = "pixai"  # "pixai" | "wd"

# Per-backend default thresholds
_DEFAULTS = {
    "pixai": (0.17, 0.27),
    "wd":    (0.35, 0.85),
}

# Pixai defaults — used to detect "caller passed pixai defaults, backend is wd"
_PIXAI_DEFAULT_GEN  = 0.17
_PIXAI_DEFAULT_CHAR = 0.27

_SUFFIXES = ["png", "webp", "jpg"]
_AUTO_TOKENS = ["animestyle", "arca_aiart_style"]


# ============================================================
# Shared utilities
# ============================================================


def get_active_token(source_name: str) -> str:
    """
    根据文件夹名自动推断 active token。
    若名称包含已知关键词则直接返回，否则提示用户输入。
    """
    name_lower = source_name.lower()
    for token in _AUTO_TOKENS:
        if token in name_lower:
            print(f"Auto active token: {token}")
            return token
    token = input(f"Active Token ({source_name}): ").strip() or source_name
    return token.lower().replace("_", " ")


# ============================================================
# Internal dispatch
# ============================================================


def _resolve_thresholds(general_threshold, character_threshold):
    """Replace the other backend's default values with the current backend's defaults."""
    other = "wd" if BACKEND == "pixai" else "pixai"
    other_gen, other_char = _DEFAULTS[other]
    cur_gen, cur_char = _DEFAULTS[BACKEND]
    if general_threshold == other_gen:
        general_threshold = cur_gen
    if character_threshold == other_char:
        character_threshold = cur_char
    return general_threshold, character_threshold


def _tag_image(image, general_threshold, character_threshold):
    general_threshold, character_threshold = _resolve_thresholds(
        general_threshold, character_threshold
    )
    if BACKEND == "pixai":
        from pixai_tagger import pixai_tag
        return pixai_tag(image, general_threshold, character_threshold)
    elif BACKEND == "wd":
        from wd_tagger import wd_eva02_tag
        return wd_eva02_tag(image, general_threshold, character_threshold)
    else:
        raise ValueError(f"Unknown backend: {BACKEND!r}. Choose 'pixai' or 'wd'.")


# ============================================================
# Public API
# ============================================================


def tag_image(
    image,
    general_threshold=0.17,
    character_threshold=0.27,
):
    """
    Tag a single image using the active backend.

    Args:
        image:
            PIL.Image.Image or image path (str / Path).

        general_threshold:
            General tag confidence threshold.

        character_threshold:
            Character tag confidence threshold.

    Returns:
        tags:
            Comma-separated tag string.

        rating:
            Dict containing rating probabilities.

        character_tags:
            Dict {tag: probability}

        general_tags:
            Dict {tag: probability}
    """
    gen, char = _resolve_thresholds(general_threshold, character_threshold)
    print(f"[tagger] backend={BACKEND}  gen={gen}  char={char}")
    return _tag_image(image, general_threshold, character_threshold)


def process_image_and_save_tags(
    image_dir: "Path | str",
    gen_threshold=0.17,
    character_threshold=0.27,
) -> "dict[Path, list[str]]":
    """
    遍历目录，对每张图片打标。

    Returns:
        dict: {image_path: [tag, ...]}
    """
    gen_threshold, character_threshold = _resolve_thresholds(gen_threshold, character_threshold)
    print(f"[tagger] backend={BACKEND}  gen={gen_threshold}  char={character_threshold}")
    image_dir = Path(image_dir)
    image_paths = []
    for suffix in _SUFFIXES:
        image_paths.extend(image_dir.glob(f"*.{suffix}"))

    results = {}
    for path in image_paths:
        tags_str, *_ = _tag_image(path, gen_threshold, character_threshold)
        results[path] = [t.strip() for t in tags_str.split(",") if t.strip()]
    return results

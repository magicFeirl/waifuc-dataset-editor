import numpy as np
import pandas as pd
import torch
import timm

from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
from timm.data import create_transform, resolve_data_config
from torch.nn import functional as F


# ============================================================
# WD-EVA02-Large v3
# ============================================================

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = timm.create_model(
    "hf-hub:" + MODEL_REPO,
    pretrained=True,
)

model = model.to(DEVICE).eval()


# ------------------------------------------------------------
# Load tags
# ------------------------------------------------------------

csv_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="selected_tags.csv",
)

tags_df = pd.read_csv(
    csv_path,
    usecols=["name", "category"],
)

tag_names = tags_df["name"].tolist()

rating_indexes = list(np.where(tags_df["category"] == 9)[0])

general_indexes = list(np.where(tags_df["category"] == 0)[0])

character_indexes = list(np.where(tags_df["category"] == 4)[0])


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

transform = create_transform(
    **resolve_data_config(
        model.pretrained_cfg,
        model=model,
    )
)


def prepare_image(image: Image.Image):
    """
    Same basic preprocessing as SmilingWolf's WD Tagger:

    RGBA -> white background
    pad to square
    model transform
    RGB -> BGR
    """

    # RGBA / RGB
    if image.mode not in ["RGB", "RGBA"]:
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")

    # Composite transparency onto white
    if image.mode == "RGBA":
        canvas = Image.new(
            "RGBA",
            image.size,
            (255, 255, 255),
        )

        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

    # Pad to square
    width, height = image.size
    max_dim = max(width, height)

    canvas = Image.new(
        "RGB",
        (max_dim, max_dim),
        (255, 255, 255),
    )

    canvas.paste(
        image,
        (
            (max_dim - width) // 2,
            (max_dim - height) // 2,
        ),
    )

    image = canvas

    # timm preprocessing
    image = transform(image)

    # RGB -> BGR
    image = image[[2, 1, 0]]

    return image.unsqueeze(0)


# ============================================================
# Main function
# ============================================================


def wd_eva02_tag(
    image,
    general_threshold=0.35,
    character_threshold=0.85,
):
    """
    WD-EVA02-Large v3 image tagging.

    Args:
        image:
            PIL.Image.Image or image path

        general_threshold:
            General tag confidence threshold.

        character_threshold:
            Character tag confidence threshold.

    Returns:
        tags:
            Final tag string.

        rating:
            Dict containing rating probabilities.

        character_tags:
            Dict {tag: probability}

        general_tags:
            Dict {tag: probability}
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image = prepare_image(image)

    image = image.to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():
        outputs = model(image)

        # timm model does not apply sigmoid
        outputs = F.sigmoid(outputs)

        outputs = outputs[0].cpu().numpy()

    # --------------------------------------------------------
    # Build label + probability pairs
    # --------------------------------------------------------

    labels = list(
        zip(
            tag_names,
            outputs,
        )
    )

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    rating = {labels[i][0]: float(labels[i][1]) for i in rating_indexes}

    # --------------------------------------------------------
    # General tags
    # --------------------------------------------------------

    general_tags = {
        labels[i][0]: float(labels[i][1])
        for i in general_indexes
        if labels[i][1] > general_threshold
    }

    # Sort by confidence
    general_tags = dict(
        sorted(
            general_tags.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # --------------------------------------------------------
    # Character tags
    # --------------------------------------------------------

    character_tags = {
        labels[i][0]: float(labels[i][1])
        for i in character_indexes
        if labels[i][1] > character_threshold
    }

    # Sort by confidence
    character_tags = dict(
        sorted(
            character_tags.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # --------------------------------------------------------
    # Combine
    #
    # Same order as the original WD Tagger:
    #
    # General first
    # Character second
    # --------------------------------------------------------

    combined_tags = list(general_tags.keys()) + list(character_tags.keys())

    # Caption:
    # Original underscore-based Danbooru tags
    caption = ", ".join(combined_tags)

    # User-facing tag string:
    # underscores -> spaces
    # escape parentheses
    tags = caption.replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return (
        tags,
        rating,
        character_tags,
        general_tags,
    )


_SUFFIXES = ["png", "webp", "jpg"]

_AUTO_TOKENS = ["animestyle", "arca_aiart_style"]


def get_active_token(source_name: str) -> str:
    """
    根据文件夹名自动推断 active token。
    若名称包含已知关键词（animestyle、arca_aiart_style）则直接返回，
    否则提示用户输入，留空时以文件夹名作为默认值。

    Returns:
        归一化后的 token（小写，下划线替换为空格）
    """
    name_lower = source_name.lower()
    for token in _AUTO_TOKENS:
        if token in name_lower:
            print(f"Auto active token: {token}")
            return token
    token = input(f"Active Token ({source_name}): ").strip() or source_name
    return token.lower().replace("_", " ")


def process_image_and_save_tags(
    image_dir: Path | str,
    gen_threshold=0.35,
    character_threshold=0.85,
) -> dict[Path, list[str]]:
    """
    遍历目录，对每张图片调用 wd_eva02_tag 打标。

    Returns:
        dict: {image_path: [tag, ...]}
    """
    image_dir = Path(image_dir)
    image_paths = []
    for suffix in _SUFFIXES:
        image_paths.extend(image_dir.glob(f"*.{suffix}"))

    results = {}
    for path in image_paths:
        tags_str, *_ = wd_eva02_tag(path, gen_threshold, character_threshold)
        results[path] = [t.strip() for t in tags_str.split(",") if t.strip()]
    return results


if __name__ == "__main__":
    images = [
        Image.open(path)
        for path in list(Path(r"E:\dataset\waifuc\vegapunk_lilith_test").glob("*.jpg"))[
            :4
        ]
    ]

    results = wd_eva02_tag(
        "E:\dataset\waifuc\vegapunk_lilith_test\1.jpg",
        general_threshold=0.35,
        character_threshold=0.85,
    )
    print(results)

from pathlib import Path
from PIL import Image
from transformers import pipeline, AutoImageProcessor

MODEL_ID = "pixai-labs/pixai-tagger-v1.0"
REVISION = '9fe10addf9326e292da8a85a98ea74cd91b41771'
_tagger = None


def _get_tagger():
    global _tagger
    if _tagger is None:
        image_processor = AutoImageProcessor.from_pretrained(
            MODEL_ID, revision=REVISION, use_fast=True, trust_remote_code=True
        )
        _tagger = pipeline(
            model=MODEL_ID,
            image_processor=image_processor,
            trust_remote_code=True,
            revision=REVISION,
        )
    return _tagger


def pixai_tag(
    image,
    general_threshold=0.17,
    character_threshold=0.27,
):
    """
    PixAI Tagger v1.0 image tagging.

    Args:
        image:
            PIL.Image.Image or image path

        general_threshold:
            General tag confidence threshold.

        character_threshold:
            Character tag confidence threshold.

    Returns:
        tags:
            Final tag string (spaces, escaped parens).

        rating:
            Dict containing rating probabilities.

        character_tags:
            Dict {tag: probability}

        general_tags:
            Dict {tag: probability}
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    results = _get_tagger()(image)["results"]

    raw_general = results.get("general", {})
    raw_character = results.get("character", {})
    rating = results.get("rating", {})

    general_tags = {k: v for k, v in raw_general.items() if v >= general_threshold}
    general_tags = dict(sorted(general_tags.items(), key=lambda x: x[1], reverse=True))

    character_tags = {k: v for k, v in raw_character.items() if v >= character_threshold}
    character_tags = dict(sorted(character_tags.items(), key=lambda x: x[1], reverse=True))

    combined = list(general_tags.keys()) + list(character_tags.keys())
    tags = ", ".join(
        t.replace("_", " ").replace("(", r"\(").replace(")", r"\)")
        for t in combined
    )

    return tags, rating, character_tags, general_tags

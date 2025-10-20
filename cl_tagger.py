import numpy as np
from PIL import Image
import json
import os
import io
import requests
import onnxruntime as ort
import time
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import Dict, Union
import gc


# --- Data Classes and Helper Functions (from original app.py) ---
@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    artist: list[np.int64]
    character: list[np.int64]
    copyright: list[np.int64]
    meta: list[np.int64]
    quality: list[np.int64]
    model: list[np.int64]


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # Converts image to RGB format if not already
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    # Pads image to be a square
    width, height = image.size
    if width == height:
        return image
    new_size = max(width, height)
    new_image = Image.new(image.mode, (new_size, new_size), (255, 255, 255))
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(image, paste_position)
    return new_image


def load_tag_mapping(mapping_path):
    # Loads tag and category information from the mapping file
    with open(mapping_path, "r", encoding="utf-8") as f:
        tag_mapping_data = json.load(f)

    idx_to_tag = {}
    tag_to_category = {}
    for k, v in tag_mapping_data.items():
        idx_to_tag[int(k)] = v["tag"]
        tag_to_category[v["tag"]] = v["category"]
    # if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
    #     idx_to_tag = {int(k): v for k, v in tag_mapping_data["idx_to_tag"].items()}
    #     tag_to_category = tag_mapping_data["tag_to_category"]
    # else:
    #     raise ValueError("Unsupported tag mapping format: Expected a dictionary with 'idx_to_tag'.")
    names = [None] * (max(idx_to_tag.keys()) + 1)
    rating, general, artist, character, copyright, meta, quality, model_name = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for idx, tag in idx_to_tag.items():
        if idx >= len(names):
            names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        category = tag_to_category.get(tag, "Unknown")
        idx_int = int(idx)
        if category == "Rating":
            rating.append(idx_int)
        elif category == "General":
            general.append(idx_int)
        elif category == "Artist":
            artist.append(idx_int)
        elif category == "Character":
            character.append(idx_int)
        elif category == "Copyright":
            copyright.append(idx_int)
        elif category == "Meta":
            meta.append(idx_int)
        elif category == "Quality":
            quality.append(idx_int)
        elif category == "Model":
            model_name.append(idx_int)

    return LabelData(
        names=names,
        rating=np.array(rating, dtype=np.int64),
        general=np.array(general, dtype=np.int64),
        artist=np.array(artist, dtype=np.int64),
        character=np.array(character, dtype=np.int64),
        copyright=np.array(copyright, dtype=np.int64),
        meta=np.array(meta, dtype=np.int64),
        quality=np.array(quality, dtype=np.int64),
        model=np.array(model_name, dtype=np.int64),
    )


def preprocess_image(image: Image.Image, target_size=(448, 448)):
    # Preprocesses a PIL image for the ONNX model
    image = pil_ensure_rgb(image)
    image = pil_pad_square(image)
    image_resized = image.resize(target_size, Image.BICUBIC)
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
    img_array = img_array[::-1, :, :]  # RGB -> BGR
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_tags(probs, labels: LabelData, gen_threshold, char_threshold):
    # Processes model output probabilities to get categorized tags
    result = {
        "rating": [],
        "general": [],
        "character": [],
        "copyright": [],
        "artist": [],
        "meta": [],
        "quality": [],
        "model": [],
    }
    # Rating and Quality (select max probability)
    for category_name, indices in [
        ("rating", labels.rating),
        ("quality", labels.quality),
    ]:
        if len(indices) > 0:
            valid_indices = indices[indices < len(probs)]
            if len(valid_indices) > 0:
                category_probs = probs[valid_indices]
                if len(category_probs) > 0:
                    max_idx_local = np.argmax(category_probs)
                    max_idx_global = valid_indices[max_idx_local]
                    if max_idx_global < len(labels.names):
                        tag_name = labels.names[max_idx_global]
                        tag_conf = float(category_probs[max_idx_local])
                        result[category_name].append((tag_name, tag_conf))

    # Other categories (based on threshold)
    category_map = {
        "general": (labels.general, gen_threshold),
        "character": (labels.character, char_threshold),
        "copyright": (labels.copyright, char_threshold),
        "artist": (labels.artist, char_threshold),
        "meta": (labels.meta, gen_threshold),
        "model": (labels.model, gen_threshold),
    }
    for category, (indices, threshold) in category_map.items():
        if len(indices) > 0:
            valid_indices = indices[indices < len(probs)]
            if len(valid_indices) > 0:
                category_probs = probs[valid_indices]
                mask = category_probs >= threshold
                selected_indices_global = valid_indices[mask]
                selected_probs = category_probs[mask]
                for idx, prob in zip(selected_indices_global, selected_probs):
                    if idx < len(labels.names):
                        result[category].append((labels.names[idx], float(prob)))

    # Sort all tags by probability
    for k in result:
        result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)
    return result


# --- Constants and Globals ---
REPO_ID = "Nonene/cl_tagger"
# model name（dir）
MODEL_OPTIONS = {
    # onnx file
    "cl_tagger_1_02": "cl_tagger_1_02/model.onnx",
}
DEFAULT_MODEL = "cl_tagger_1_02"
CACHE_DIR = "./model_cache"

# Global state variables
g_onnx_model_path = None
g_labels_data = None
g_current_model = None
g_session = None


def release_model():
    """Releases the loaded model and frees up memory."""
    global g_onnx_model_path, g_labels_data, g_current_model, g_session

    if g_session is not None:
        print(f"Releasing model: {g_current_model}...")

        # 核心步骤：将 session 和其他数据设置为 None
        g_session = None
        g_labels_data = None
        g_current_model = None
        g_onnx_model_path = None

        # 建议：显式调用垃圾回收
        # 这会提示 Python 尽快回收不再使用的对象所占用的内存（特别是 GPU 显存）
        gc.collect()

        print("Model released successfully. ✅")
    else:
        print("No model is currently loaded.")


# --- Core Logic ---
def initialize_model(model_choice=DEFAULT_MODEL):
    """Downloads model files and loads labels into global state."""
    global g_onnx_model_path, g_labels_data, g_current_model, g_session

    if g_current_model == model_choice and g_session is not None:
        # print(f"Model '{model_choice}' is already initialized.")
        return

    if model_choice not in MODEL_OPTIONS:
        raise ValueError(
            f"Invalid model choice: {model_choice}. Available: {list(MODEL_OPTIONS.keys())}"
        )

    print(f"Initializing model: {model_choice}...")
    model_dir = model_choice
    onnx_filename = MODEL_OPTIONS[model_choice]
    tag_mapping_filename = f"{model_dir}/tag_mapping.json"
    hf_token = os.environ.get("HF_TOKEN")

    g_onnx_model_path = hf_hub_download(
        repo_id=REPO_ID, filename=onnx_filename, cache_dir=CACHE_DIR, token=hf_token
    )
    tag_mapping_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=tag_mapping_filename,
        cache_dir=CACHE_DIR,
        token=hf_token,
    )
    g_labels_data = load_tag_mapping(tag_mapping_path)

    # Load ONNX session
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    g_session = ort.InferenceSession(g_onnx_model_path, providers=providers)

    g_current_model = model_choice
    print(
        f"Model '{model_choice}' loaded successfully using: {g_session.get_providers()[0]}"
    )


def get_tags_for_image(
    image_input: Union[str, np.ndarray, Image.Image],
    model_choice: str = DEFAULT_MODEL,
    gen_threshold: float = 0.55,
    char_threshold: float = 0.60,
) -> Dict:
    """
    Predicts tags for a given image using the specified ONNX model.

    Args:
        image_input: The input image. Can be a URL (str), file path (str),
                     NumPy array, or PIL Image.
        model_choice: The name of the model to use.
        gen_threshold: Threshold for general, meta, and model tags.
        char_threshold: Threshold for character, copyright, and artist tags.

    Returns:
        A dictionary containing the predicted tags, categorized and sorted.
    """
    # --- 1. Initialize model if necessary ---
    initialize_model(model_choice)

    # --- 2. Process Input Image ---
    try:
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.astype(np.float32)

    except Exception as e:
        raise IOError(f"Error processing input image: {e}") from e

    # --- 3. Run Inference ---
    try:
        input_name = g_session.get_inputs()[0].name
        output_name = g_session.get_outputs()[0].name

        start_time = time.time()
        outputs = g_session.run([output_name], {input_name: input_tensor})[0]
        # print(f"Inference completed in {time.time() - start_time:.3f} seconds")

        # Sigmoid activation to get probabilities
        probs = 1 / (1 + np.exp(-outputs[0]))

    except Exception as e:
        raise RuntimeError(f"Error during ONNX inference: {e}") from e

    # --- 4. Post-process and return results ---
    predictions = get_tags(probs, g_labels_data, gen_threshold, char_threshold)
    return predictions


def format_tags_to_string(
    predictions: Dict, categories=None, rating=False, quality=False
) -> str:
    """Converts the prediction dictionary into a comma-separated string."""
    output_tags = []
    # Add rating and quality first
    if rating and predictions.get("rating"):
        output_tags.append(predictions["rating"][0][0].replace("_", " "))
    if quality and predictions.get("quality"):
        output_tags.append(predictions["quality"][0][0].replace("_", " "))

    if not categories:
        categories = ["artist", "character", "copyright", "general", "meta", "model"]
    # Add other categories
    for category in categories:
        for tag, prob in predictions.get(category, []):
            output_tags.append(tag.replace("_", " "))

    return ", ".join(output_tags)


def process_image_and_save_tags(
    image_path,
    gen_threshold=0.55,
    char_threshold=0.80,
):
    try:
        predicted_tags_dict = get_tags_for_image(
            image_input=image_path,
            model_choice=DEFAULT_MODEL,
            gen_threshold=gen_threshold,
            char_threshold=char_threshold,
        )

        return format_tags_to_string(predicted_tags_dict, categories=["general"])
    except (IOError, RuntimeError, ValueError) as e:
        print(f"An error occurred while processing '{image_path}': {e}")

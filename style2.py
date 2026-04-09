# @title Waifuc + Wildcard Extractor (Dict & Natural Sort Version)
import argparse
import random
import shutil
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

# --- Waifuc 相关导入 ---
from waifuc.action import (
    ModeConvertAction,
    FirstNSelectAction,
    FilterSimilarAction,
    FileOrderAction,
    FileExtAction,
    HeadCountAction,
    NoMonochromeAction,
)
from waifuc.export import TextualInversionExporter
from waifuc.source import LocalSource

# --- Tagger 导入 ---
from cl_tagger import process_image_and_save_tags


def banner(message):
    print("*" * 40)
    print(f" {message}")
    print("*" * 40)
    print()


# ==========================================
# 0. 工具函数：自然排序
# ==========================================
def natural_sort_key(path_obj: Path):
    """
    用于自然排序的 key 函数。
    例如让 2.jpg 排在 10.jpg 前面，而不是 10.jpg, 2.jpg
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', path_obj.name)]


# ==========================================
# 1. 字典加载与过滤逻辑
# ==========================================
TAG_DICT = {}
DICT_PATH = Path("tags.txt")

# 尝试加载外部的 tags.txt 字典文件
if DICT_PATH.exists():
    try:
        # 使用 utf-8-sig 兼容带 BOM 的文件
        raw_dict = json.loads(DICT_PATH.read_text(encoding="utf-8-sig"))
        # 🔥 关键修复：将字典键中的下划线统一替换为空格，匹配 Tagger 输出
        TAG_DICT = {k.replace("_", " "): v for k, v in raw_dict.items()}
        print(f"✅ 成功加载了 {len(TAG_DICT)} 个 Tag 分类规则字典 (已处理下划线)！")
    except Exception as e:
        print(f"❌ 解析 tags.txt 失败，将使用基础黑名单过滤: {e}")
else:
    print("⚠️ 未找到 tags.txt，将使用基础黑名单规则进行 Wildcard 提取。")

# 基础黑名单（作为字典缺失时的兜底）
EXACT_BLACKLIST = {
    "breasts", "medium breasts", "large breasts", "small breasts", "huge breasts",
    "cleavage", "collarbone", "thighs", "bare shoulders", "navel", "armpit crease",
    "covering privates", "nude cover", "naked towel", "bangs", "blunt bangs", 
    "messy hair", "ahoge", "twintails", "ponytail", "braid", "virtual youtuber", 
    "vocaloid", "1girl", "2girls", "3girls", "1boy", "2boys", "solo"
}

def is_character_feature(tag: str) -> bool:
    """
    判断该 tag 是否是人物特征/服装/无效元数据，返回 True 代表需要剔除
    """
    clean_tag = tag.lower().strip().replace("_", " ")
    words = clean_tag.split()
    
    # ==========================================
    # 1. 最高优免死金牌 (动作与身体/NSFW特征)
    # 必须放在最前面！只要包含这些词，不管字典里怎么说，绝对保留！
    # ==========================================
    
    # 动作白名单 (包含常规动作与部分NSFW互动动作)
    action_verbs = [
        "holding", "pulling", "adjusting", "removing", "lifting", 
        "grabbing", "touching", "hand on", "feet on", "hand in", "looking at",
        "pointing", "reaching", "biting", "licking", "kissing", "hugging",
        "crossed", "leaning", "sitting", "standing", "lying", "kneeling",
        "looking", "pose", "gesture", "straddling", "grinding", "spreading", 
        "rubbing", "inserting", "penetration", "fingering"
    ]
    
    # 🔥 NSFW 与 身体结构白名单 (覆盖 90% 以上场景)
    anatomy_keywords = [
        # --- 胸部与乳房 (Breasts) ---
        "breast", "breasts", "cleavage", "nipple", "nipples", "areola", "areolae", 
        "underboob", "sideboob", "boob", "boobs", "paizuri",
        
        # --- 腹部与下半身 (Abdomen & Lower Body) ---
        "navel", "stomach", "belly", "groin", "crotch", "thigh", "thighs", "hip", "hips",
        
        # --- 女性私密部位 (Female Genitals) ---
        "pussy", "vulva", "cameltoe", "clitoris", "labia", "pubic", "slit", "mound of venus",
        
        # --- 臀部与后庭 (Ass & Anus) ---
        "ass", "butt", "buttocks", "anus", "asshole", "anal", "crack", "gluteal fold",
        
        # --- 男性私密部位 (Male Genitals - 如果你不跑扶他或男性可以删掉这行) ---
        "penis", "cock", "dick", "testicles", "balls", "foreskin", "erection", "shaft",
        
        # --- 体液与分泌物 (Fluids & Secretions) ---
        "cum", "semen", "squirt", "juice", "sweat", "saliva", "drool", "tears", "urine", "pee", "wet",
        
        # --- NSFW 状态与表情 (NSFW States & Expressions) ---
        "nude", "naked", "bare", "exposed", "uncensored", "orgasm", "ahegao", "blush",
        "creampie", "bukkake", "mind break", "messy", "dirty",
        
        # --- NSFW 行为/流派名词 (NSFW Acts/Tags) ---
        "sex", "fellatio", "cunnilingus", "masturbation", "doggystyle", "missionary", 
        "cowgirl", "69", "tribadism", "tentacle", "bondage", "tied up"
    ]
    
    # 检查是否命中动作 (动词可以用普通的包含匹配，因为大多是ing后缀，比较安全)
    for action in action_verbs:
        if action in clean_tag:
            return False
            
    # 🔥 检查是否命中身体部位 (使用正则单词边界 \b，防止 ass 匹配到 glasses)
    for anatomy in anatomy_keywords:
        # \b 代表单词边界，这意味着 "ass" 只会匹配 "ass" 或 "bare ass"，不会匹配 "glasses"
        if re.search(r'\b' + re.escape(anatomy) + r'\b', clean_tag):
            return False

    # ==========================================
    # 2. 精确匹配字典
    # ==========================================
    if TAG_DICT and clean_tag in TAG_DICT:
        category = TAG_DICT[clean_tag]
        if category in ["appearance", "clothing", "other", "0"]:
            return True
        if category in ["pose", "scenery"]:
            return False

    # ==========================================
    # 3. 基础黑名单与器官特征兜底
    # 注意：我已经在 EXACT_BLACKLIST 中去除了 breast, thighs 等词
    # ==========================================
    EXACT_BLACKLIST = {
        "collarbone", "bare shoulders", "armpit crease", "covering privates", 
        "bangs", "blunt bangs", "messy hair", "ahoge", "twintails", "ponytail", 
        "braid", "virtual youtuber", "vocaloid", "1girl", "2girls", "3girls", 
        "1boy", "2boys", "solo"
    }
    if clean_tag in EXACT_BLACKLIST:
        return True
        
    if "hair" in clean_tag:
        return True
    if "eyes" in clean_tag:
        keep_eyes = ["closed eyes", "half-closed eyes", "rolling eyes", "crazy eyes"]
        if clean_tag not in keep_eyes:
            return True

    # ==========================================
    # 4. 模糊匹配 (前后缀剥离查字典)
    # ==========================================
    if TAG_DICT and len(words) > 1:
        # 查后缀
        for i in range(1, len(words)):
            suffix = " ".join(words[i:])
            if suffix in TAG_DICT:
                cat = TAG_DICT[suffix]
                if cat in ["appearance", "clothing", "0", "other"]:
                    return True
                if cat in ["pose", "scenery"]:
                    return False 
                    
        # 查前缀
        for i in range(len(words)-1, 0, -1):
            prefix = " ".join(words[:i])
            if prefix in TAG_DICT:
                cat = TAG_DICT[prefix]
                if cat in ["appearance", "clothing", "0", "other"]:
                    return True
                if cat in ["pose", "scenery"]:
                    return False

    # ==========================================
    # 5. 颜色词暴力截杀
    # ==========================================
    colors = {
        "white", "black", "red", "blue", "green", "yellow", "purple", "pink", 
        "orange", "brown", "grey", "gray", "silver", "gold", "blonde", "cyan", "magenta"
    }
    if any(color in words for color in colors):
        return True

    # ==========================================
    # 6. 核心名词兜底 (极大扩充了配饰和衣服)
    # ==========================================
    core_nouns = [
        "shirt", "skirt", "dress", "jacket", "coat", "pants", "shorts", 
        "panties", "underwear", "bra", "socks", "shoes", "boots", "gloves", 
        "towel", "hat", "headwear", "pantyhose", "bow", "ribbon", "tie", 
        "tattoo", "mark", "symbol", "gemstone", "sash", "uniform", "pouch",
        "collar", "necklace", "earrings", "ring", "bracelet", "glasses",
        "bikini", "swimsuit", "leotard", "band", "clip", "pin", "accessory",
        "choker", "stockings", "tights", "leggings", "cape", "cloak",
        "apron", "scarf", "jewel", "ornament"
    ]
    for noun in core_nouns:
        if noun in words:
            return True

    # 默认保留
    return False

# ==========================================
# 2. 纯提取 Wildcard 模式 (不输出图片)
# ==========================================
def extract_wildcards_from_album(album_path: Path):
    if not album_path.is_dir():
        print(f"Error: 找不到图集文件夹 {album_path}")
        return

    banner(f"🔍 正在提取 Wildcard 图集: {album_path.name}")
    image_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    
    images = []
    for ext in image_exts:
        images.extend(album_path.rglob(f"*{ext}"))
        
    # 🔥 关键修复：应用自然排序
    images.sort(key=natural_sort_key)

    if not images:
        print("未在该目录下找到任何图片！")
        return

    print(f"✅ 找到 {len(images)} 张图片，开始剥离人物特征...")
    wildcards = []

    for idx, img_path in enumerate(images):
        print(f"  [{idx+1}/{len(images)}] 处理中: {img_path.name}")
        
        raw_tags_str = process_image_and_save_tags(
            image_path=str(img_path),
            gen_threshold=0.45,
        )
        
        tags = [t.strip() for t in raw_tags_str.split(",") if t.strip()]
        cleaned_tags = [t for t in tags if not is_character_feature(t)]
        
        wildcard_line = ", ".join(cleaned_tags)
        if wildcard_line:
            wildcards.append(wildcard_line)

    output_file = Path.cwd() / f"{album_path.name}_wildcards.txt"
    output_file.write_text("\n".join(wildcards), encoding="utf-8")
    
    print("\n🎉 提取完成！")
    print(f"📁 Wildcard 动作集已保存至: {output_file.absolute()}")


# ==========================================
# 3. Waifuc 图片处理与打标抽样模式
# ==========================================
def run_local_source(source: str, dest: str, max_count: int = 180):
    (LocalSource(source)).attach(
        ModeConvertAction("RGB", "white"),
        NoMonochromeAction(),
        HeadCountAction(min_count=1),
        FileOrderAction(),
        FileExtAction(ext=".jpg"),
        FirstNSelectAction(max_count)
    ).export(TextualInversionExporter(dest))
    return dest.absolute()


def sample_and_stage_images(root_dir: Path, target_n: int) -> Path:
    print(f"Scanning for images in {root_dir}...")
    image_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    folder_to_images = defaultdict(list)

    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            folder_to_images[file_path.parent].append(file_path)

    if not folder_to_images:
        print("Error: No images found in the directory!")
        return None

    for folder in folder_to_images:
        random.shuffle(folder_to_images[folder])

    sampled_images = []
    active_folders = list(folder_to_images.keys())
    random.shuffle(active_folders)

    while len(sampled_images) < target_n and active_folders:
        for folder in list(active_folders):
            if len(sampled_images) >= target_n:
                break
            if folder_to_images[folder]:
                sampled_images.append(folder_to_images[folder].pop(0))
            else:
                active_folders.remove(folder)

    staging_dir = Path("./temp_staging_dir") / root_dir.name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(sampled_images):
        dest_path = staging_dir / f"sampled_{idx:04d}{img_path.suffix}"
        shutil.copy2(img_path, dest_path)

    print(f"✅ Successfully sampled {len(sampled_images)} images from {len(folder_to_images)} folders.")
    return staging_dir


def waifuc(path: Path, output_name_override: str = None):
    iterdir = [n for n in path.iterdir() if n.is_dir()]
    if len(iterdir) == 0:
        iterdir = [path]

    for source in iterdir:
        if not source.is_dir():
            continue

        base_name = output_name_override if output_name_override else source.name.split("-")[0]
        dest: Path = Path("./output/") / base_name

        if not dest.is_dir():
            print("Processing:", source)
            run_local_source(source, dest)
        else:
            print(f"{dest} existed, skipping waifuc filtering")

        active_tokens = input(f"Active Tokens for '{base_name}': ")
        if not active_tokens:
            active_tokens = base_name

        shuffix = ["png", "webp", "jpg"]
        files = []
        for s in shuffix:
            files.extend(Path(dest).glob(f"*.{s}"))
            
        # 🔥 在打标时也应用自然排序
        files.sort(key=lambda x: natural_sort_key(Path(x)))

        print(f"Tagging {len(files)} images...")
        for image_path in files:
            filename = Path(image_path).with_suffix(".txt")
            tags = process_image_and_save_tags(
                image_path=str(image_path),
                gen_threshold=0.45,
            )
            tags = [active_tokens, tags]
            filename.write_text(", ".join(tags))

        print("🎉 Output Dir:")
        print(dest.absolute())


# ==========================================
# 4. 主程序入口与命令行解析
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waifuc Dataset Builder & Wildcard Extractor")
    parser.add_argument("target", nargs="?", default="", help="Target directory to process")
    parser.add_argument("-n", "--sample", type=int, default=0, help="Randomly sample N images evenly across subfolders")
    parser.add_argument("-w", "--wildcard", action="store_true", help="Extract tags as a wildcard file (no image output)")
    
    args = parser.parse_args()
    target = args.target

    # ---- 命令行自动模式 ----
    if target:
        target_path = Path(target)
        
        if args.wildcard:
            extract_wildcards_from_album(target_path)
            sys.exit(0)
            
        if args.sample > 0:
            banner(f"Mode: Sampling {args.sample} images")
            staging_path = sample_and_stage_images(target_path, args.sample)
            if staging_path:
                waifuc(staging_path, output_name_override=target_path.name)
        else:
            banner("Mode: Full Dataset Processing")
            waifuc(target_path)
            
        print("\nAll done!")
        sys.exit(0)

    # ---- 终端交互模式 ----
    while True:
        print("\n" + "="*40)
        target = input("Input Dir (or press Enter to exit): ").strip()
        if not target:
            break
        
        target_path = Path(target)
        mode = input("Select mode: [1] 常规处理打标 [2] 提取动作/场景 Wildcard: ").strip()
        
        if mode == "2":
            extract_wildcards_from_album(target_path)
        else:
            sample_input = input("Sample N images? (直接回车全量处理，或输入数字): ").strip()
            if sample_input.isdigit() and int(sample_input) > 0:
                staging_path = sample_and_stage_images(target_path, int(sample_input))
                if staging_path:
                    waifuc(staging_path, output_name_override=target_path.name)
            else:
                waifuc(target_path)
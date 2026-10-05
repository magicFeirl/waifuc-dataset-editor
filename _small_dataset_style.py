# @title Waifuc
from pathlib import Path
import sys
import shutil
from actions import TransformImages

from waifuc.action import (
    ModeConvertAction,
    FirstNSelectAction,
    FileOrderAction,
    FileExtAction,
    HeadCountAction,
    NoMonochromeAction,
    MinAreaFilterAction,
    ThreeStageSplitAction,
    FilterSimilarAction,
    PersonSplitAction,
)

from waifuc.export import SaveExporter
from waifuc.source import LocalSource

from tagger import process_image_and_save_tags, get_active_token


def banner(message):
    print("*" * 20)
    print(message)
    print("*" * 20)
    print()


def run_local_source(source: str, dest: str, transform_image=True):
    if transform_image:
        print("Filpping Dataset")
        TransformImages(source)

    (LocalSource(str(source), recursive=False)).attach(
        # RandomChoiceAction(p=0.3),
        ModeConvertAction("RGB", "white"),
        # NoMonochromeAction(),
        # FilterSimilarAction(threshold=0.45),
        HeadCountAction(min_count=1),
        # PersonSplitAction(),
        # ThreeStageSplitAction(),
        MinAreaFilterAction(700),
        # FilterSimilarAction(threshold=0.45),
        # HeadCountAction(min_count=1),
        # FileOrderAction(),
        FileExtAction(ext=".webp"),
        FirstNSelectAction(100),
    ).export(SaveExporter(dest, no_meta=True))

    return dest.absolute()


# real life
"""
(LocalSource(str(source), recursive=False)).attach(
    # RandomChoiceAction(p=0.3),
    ModeConvertAction("RGB", "white"),
    # NoMonochromeAction(),
    FilterSimilarAction(threshold=0.45),
    HeadCountAction(min_count=1),
    ThreeStageSplitAction(),
    MinAreaFilterAction(700),
    FilterSimilarAction(threshold=0.45),
    HeadCountAction(min_count=1),
    # FileOrderAction(),
    FileExtAction(ext=".webp"),
    FirstNSelectAction(100),
).export(SaveExporter(dest, no_meta=True))
"""


def waifuc(path: str):
    path: Path = Path(path)

    # 检查是否是不含子文件夹的根文件夹
    iterdir = [n for n in path.iterdir() if n.is_dir()]
    if len(iterdir) == 0:
        iterdir = [path]

    for source in iterdir:
        if not source.is_dir():
            continue

        dest: Path = Path("./output/") / source.name

        if dest.is_dir():
            print("Rm:", dest)
            shutil.rmtree(dest, ignore_errors=True)

        file_count = len(list(Path(source).glob("*")))
        transform_image = file_count <= 20
        print("Processing:", source)
        run_local_source(source, dest, transform_image)

        active_tokens = get_active_token(source.name)

        all_tags = set()
        tagged = process_image_and_save_tags(image_dir=dest, gen_threshold=0.35)
        for image_path, tags in tagged.items():
            filename = image_path.with_suffix(".txt")
            all_tags.update(tags)
            tags = [tag.lower() for tag in tags if tag not in active_tokens]
            tags.insert(0, active_tokens)
            if "_flip" in filename.name and "upside-down" not in tags and transform_image:
                tags.append("upside-down")
            filename.write_text(", ".join(tags))

        if all_tags:
            checklist = ["looking", "facing", "from", "straight-on"]
            print("Check Tags:")
            filtered_tags = [
                tag
                for tag in all_tags
                if any(ck in tag for ck in checklist)
            ]
            filtered_tags.sort(key=lambda k: len(k), reverse=True)
            print("\n".join(filtered_tags))

        print(f"Output Dir ({len(tagged)} files):")
        print(dest.absolute())


if __name__ == "__main__":
    target = r""

    if len(sys.argv) == 2:
        target = sys.argv[1]
    else:
        target = input("Input Dir:")

    while target:
        waifuc(target)
        print()
        target = input("Input Dir:")
        print()

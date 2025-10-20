# @title Waifuc
from pathlib import Path
import sys

from waifuc.action import (
    ModeConvertAction,
    ThreeStageSplitAction,
    CCIPAction,
    FilterSimilarAction,
    FileOrderAction,
    FileExtAction,
)

from waifuc.export import TextualInversionExporter
from waifuc.source import LocalSource

from cl_tagger import process_image_and_save_tags
from tag_cleanr import TagCleaner


def banner(message):
    print("*" * 20)
    print(message)
    print("*" * 20)
    print()


def run_local_source(source: str, dest: str):
    (LocalSource(source)).attach(
        ModeConvertAction("RGB", "white"),
        ThreeStageSplitAction(),
        CCIPAction(min_val_count=15),
        FilterSimilarAction(),
        FileOrderAction(),
        # TaggingAction(),
        FileExtAction(ext=".png"),
    ).export(TextualInversionExporter(dest))

    return dest.absolute()


def waifuc(path: str):
    # [r"E:\dataset\2025年10月12日\bili_girl_22_dress"]:
    path: Path = Path(path)

    # 检查是否是不含子文件夹的根文件夹
    iterdir = [n for n in path.iterdir() if n.is_dir()]
    if len(iterdir) == 0:
        iterdir = [path]

    for source in iterdir:
        if not source.is_dir():
            continue

        dest: Path = Path("./output/") / (source.name.split('-')[0] + "_waifuc")
        if not dest.is_dir():
            print("Processing:", source)
            run_local_source(source, dest)
        else:
            print(f'{dest} existed, skipping waifuc')

        tag_cleaner = TagCleaner()
        for image_path in Path(dest).glob("*.png"):
            filename = Path(image_path).with_suffix("")

            tags = process_image_and_save_tags(
                image_path=str(image_path),
                gen_threshold=0.60,
            )

            tag_cleaner.add_tags(filename=filename, tags=tags)

        total_images = tag_cleaner.size
        banner(f"{dest}: {total_images} image tagged")
        # remove top 70% tags common and in blacklisted
        for file, tags in tag_cleaner.get_cleaned_tags(round(total_images * 0.3)):
            file.with_suffix(".txt").write_text(', '.join(tags))

        print('Output Dir:')
        print(dest.absolute())
        
if __name__ == '__main__':
    target = r''

    if len(sys.argv) == 2:
        target = sys.argv[1]

    if target:
        waifuc(target)
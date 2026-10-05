# 人物 waifuc

from pathlib import Path
import sys
import shutil

from actions import TransformImages

from waifuc.action import (
    ModeConvertAction,
    FileOrderAction,
    FileExtAction,
    FirstNSelectAction,
)

from waifuc.export import TextualInversionExporter
from waifuc.source import LocalSource

from tagger import process_image_and_save_tags
from tag_cleaner import TagCleaner


def banner(message):
    print("*" * 20)
    print(message)
    print("*" * 20)
    print()


def run_local_source(source: str, dest: str):
    # print('Fliping images')
    # TransformImages(source)

    (LocalSource(source)).attach(
        ModeConvertAction("RGB", "white"),
        FileOrderAction(),
        FileExtAction(ext=".webp"),
        FirstNSelectAction(180),
    ).export(TextualInversionExporter(dest))

    return dest.absolute()


def waifuc(path: str):
    path: Path = Path(path)

    # 检查是否是不含子文件夹的根文件夹
    iterdir = [n for n in path.iterdir() if n.is_dir()]
    if len(iterdir) == 0:
        iterdir = [path]

    for source in iterdir:
        if not source.is_dir():
            continue

        dest: Path = Path("./output/") / (source.name.split("-")[0])
        if dest.is_dir():
            print("Delete existed dir:", dest)
            shutil.rmtree(dest)
        print("Processing:", source)
        run_local_source(source, dest)
        # else:
        #     print(f'{dest} existed, skipping waifuc')

        tag_cleaner = TagCleaner()
        tagged = process_image_and_save_tags(image_dir=dest, gen_threshold=0.35)
        for image_path, tags in tagged.items():
            tag_cleaner.add_tags(filename=image_path.with_suffix(""), tags=tags)

        total_images = tag_cleaner.size
        banner(f"{dest}: {total_images} image tagged")
        # remove top 70% tags common and in blacklisted
        for file, tags in tag_cleaner.get_cleaned_tags(round(total_images * 0.3)):
            file.with_suffix(".txt").write_text(", ".join(tags))

        print(f"Output Dir({tag_cleaner.file_count} images):")
        print(
            dest.name,
            f"Images Count: {total_images}. Suggest Steps: {total_images * 10 + 200}",
        )
        print(dest.absolute())


if __name__ == "__main__":
    target = r""

    if len(sys.argv) == 2:
        target = sys.argv[1]

    while target:
        waifuc(target)
        print()
        target = input("Input Dir:")

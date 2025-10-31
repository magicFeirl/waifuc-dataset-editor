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
        FileOrderAction(),
        # TaggingAction(),
        FileExtAction(ext=".jpg"),
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

        dest: Path = Path("./output/") / (source.name.split('-')[0] + "_waifuc")
        if not dest.is_dir():
            print("Processing:", source)
            run_local_source(source, dest)
        else:
            print(f'{dest} existed, skipping waifuc')

        active_tokens = input(f'Active Tokens({source.name}):')
        if not active_tokens:
            active_tokens = source.name

        shuffix = ['png', 'webp', 'jpg']
        files = []
        for s in shuffix:
            files.extend(Path(dest).glob(f"*.{s}"))

        for image_path in files:
            filename = Path(image_path).with_suffix(".txt")

            tags = process_image_and_save_tags(
                image_path=str(image_path),
                gen_threshold=0.60,
            )

            tags = [active_tokens, tags]
            filename.write_text(', '.join(tags))

        print('Output Dir:')
        print(dest.absolute())
        
if __name__ == '__main__':
    target = r''

    if len(sys.argv) == 2:
        target = sys.argv[1]

    while target:
        waifuc(target)
        print()
        target = input('Input Dir:')
        print()
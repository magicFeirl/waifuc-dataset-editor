# @title Waifuc
from pathlib import Path
import sys
import shutil

from waifuc.action import (
    ModeConvertAction,
    FirstNSelectAction,
    FileOrderAction,
    FilterSimilarAction,
    FileExtAction,
    HeadCountAction,
    NoMonochromeAction,
    MinAreaFilterAction,
    PersonRatioAction,
    ClassFilterAction,
    PersonSplitAction,
    ThreeStageSplitAction,
)

from waifuc.export import SaveExporter
from waifuc.source import LocalSource

from tagger import process_image_and_save_tags, get_active_token


def banner(message):
    print("*" * 20)
    print(message)
    print("*" * 20)
    print()


def run_local_source(source: str, dest: str):
    (LocalSource(str(source), recursive=False)).attach(
        # MinAreaFilterAction(768),
        # RandomChoiceAction(p=0.3),
        ModeConvertAction("RGB", "white"),
        NoMonochromeAction(),
        # ClassFilterAction(["illustration", "bangumi", "3d"]),
        FilterSimilarAction(threshold=0.45),  # threshold <= 0.45 可以被认为是相像的
        PersonSplitAction(),
        # HeadCountAction(min_count=1),
        ThreeStageSplitAction(),
        MinAreaFilterAction(768),
        PersonRatioAction(),
        FilterSimilarAction(threshold=0.45),  # threshold <= 0.45 可以被认为是相像的
        FileOrderAction(),
        FileExtAction(ext=".jpg"),
        FirstNSelectAction(200),
    ).export(SaveExporter(dest, no_meta=True)) # site-packages\waifuc\model\item.py L93 删除了 save_params 参数

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

        dest: Path = Path("./output/") / source.name

        if dest.is_dir():
            print('Rm:', dest)
            shutil.rmtree(dest, ignore_errors=True)

        print("Processing:", source)
        run_local_source(source, dest)

        active_tokens = get_active_token(source.name)

        tagged = process_image_and_save_tags(image_dir=dest, gen_threshold=0.35)
        for image_path, tags in tagged.items():
            filename = image_path.with_suffix(".txt")
            tags = [tag.lower() for tag in tags if tag not in active_tokens]
            tags.insert(0, active_tokens)
            filename.write_text(", ".join(tags))

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

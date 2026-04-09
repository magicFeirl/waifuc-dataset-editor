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
    ClassFilterAction
)

from waifuc.export import SaveExporter
from waifuc.source import LocalSource

from cl_tagger import process_image_and_save_tags


def banner(message):
    print("*" * 20)
    print(message)
    print("*" * 20)
    print()


def run_local_source(source: str, dest: str):
    (LocalSource(str(source), recursive=False)).attach(
        MinAreaFilterAction(768),
        # RandomChoiceAction(p=0.3),
        ModeConvertAction("RGB", "white"),
        NoMonochromeAction(),
        ClassFilterAction(["illustration", "bangumi"]),
        HeadCountAction(min_count=1),
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
        # if dest.is_dir() or len(list(dest.glob("*.*"))):
        #     old_dest_new_name = dest.with_name(dest.name + "_1")
        #     banner(f"{dest} existed, rename to {old_dest_new_name}")
        #     os.rename(dest, old_dest_new_name)

        print("Processing:", source)
        run_local_source(source, dest)

        active_tokens = input(f"Active Tokens({source.name}):")

        if not active_tokens:
            active_tokens = source.name

        active_tokens = active_tokens.lower()

        shuffix = ["png", "webp", "jpg"]
        files = []
        for s in shuffix:
            files.extend(Path(dest).glob(f"*.{s}"))

        for image_path in files:
            filename = Path(image_path).with_suffix(".txt")

            tags = process_image_and_save_tags(
                image_path=str(image_path),
                gen_threshold=0.45,
            )
            
            tags = [tag.lower() for tag in tags if tag not in active_tokens]
            tags.insert(0, active_tokens)

            filename.write_text(", ".join(tags))

        print(f"Output Dir ({len(files)} files):")
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

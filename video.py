# @title Waifuc
from pathlib import Path
import sys
import functools

from waifuc.action import (
    FirstNSelectAction,
    FilterSimilarAction,
    FileExtAction,
    FaceCountAction,
    NoMonochromeAction,
    ClassFilterAction,
    PersonRatioAction
)

from waifuc.export import SaveExporter
from waifuc.source import VideoSource


def main(source: str, dest: str):
    videos = list(Path(source).glob("*"))

    if not videos:
        videos = [Path(source)]

    print("Processing:", ", ".join([v.name for v in videos]))

    s = functools.reduce(lambda a, b: a + b, [VideoSource(str(video)) for video in videos])

    (s).attach(
        NoMonochromeAction(),
        ClassFilterAction(["illustration", "bangumi"]),
        # HeadCountAction(min_count=1),
        FaceCountAction(min_count=1, max_count=3),
        PersonRatioAction(ratio=0.6), # 提取番剧时打开
        # PersonSplitAction(),
        FilterSimilarAction(threshold=0.45),  # threshold <= 0.45 可以被认为是相像的
        # FileOrderAction(),
        FileExtAction(ext=".png"),
        FirstNSelectAction(200),
    ).export(SaveExporter(dest, no_meta=True))

    return Path(dest).absolute()


if __name__ == "__main__":
    target = r""

    if len(sys.argv) == 2:
        target = sys.argv[1]
    else:
        target = input("Input Dir:")

    while target:
        output_dir = main(source=target, dest="./video-output")
        print(output_dir)
        target = input("Input Dir:")
        print()

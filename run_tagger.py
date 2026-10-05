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


from tagger import process_image_and_save_tags, get_active_token


def banner(message):
    print("*" * 20)
    print(message)
    print("*" * 20)
    print()


def run_tagger(path: str):
    path: Path = Path(path)
    use_active_token = True

    iterdir = [n for n in path.iterdir() if n.is_dir()]

    if len(iterdir) == 0:
        iterdir = [path]

    for source in iterdir:
        dest: Path = source

        active_tokens = [get_active_token(source.name)] if use_active_token else []

        tagged = process_image_and_save_tags(image_dir=dest, gen_threshold=0.35)
        for image_path, tags in tagged.items():
            filename = image_path.with_suffix(".txt")
            tags = [*active_tokens, *tags]
            filename.write_text(', '.join(tags))

        print('Output Dir:')
        print(dest.absolute())
        
if __name__ == '__main__':
    target = r''

    if len(sys.argv) == 2:
        target = sys.argv[1]

    while target:
        run_tagger(target)
        print()
        target = input('Input Dir:')
        print()
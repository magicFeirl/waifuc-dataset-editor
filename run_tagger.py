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

        shuffix = ['png', 'webp', 'jpg']

        files = []
        for s in shuffix:
            files.extend(Path(dest).glob(f"*.{s}"))

        if use_active_token:
            active_tokens = input(f'Active Token:({source.name})')
            if not active_tokens:
                active_tokens = source.name
            active_tokens = active_tokens.split(',')
        else:
            active_tokens = []

        for image_path in files:
            filename = Path(image_path).with_suffix(".txt")

            tags = process_image_and_save_tags(
                image_path=str(image_path),
                gen_threshold=0.45,
            )

            tags = [*active_tokens, tags]
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
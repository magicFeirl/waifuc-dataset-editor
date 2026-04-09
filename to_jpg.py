from waifuc.action import (
    ModeConvertAction,
    FileOrderAction,
    FileExtAction,
)

from waifuc.export import TextualInversionExporter
from waifuc.source import LocalSource


def banner(message):
    print("*" * 20)
    print(message)
    print("*" * 20)
    print()


def run_local_source(source: str, dest: str):
    (LocalSource(source)).attach(
        ModeConvertAction("RGB", "white"),
        FileOrderAction(),
        FileExtAction(ext=".jpg"),
    ).export(TextualInversionExporter(dest))

    return dest


if __name__ == '__main__':
    run_local_source(r'D:\sd-aki\dataset_process\output\necro_complex\autopsy', r'./output/autopsy_jpg')
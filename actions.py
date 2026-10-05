import copy
from typing import Union

from PIL import ImageOps, Image

from pathlib import Path

from waifuc.action import BaseAction
from waifuc.model import ImageItem


def TransformImages(source):
    for file in Path(source).glob("*"):
        try:
            # 跳过已经翻转的图片
            save_file_path = Path(source) / (Path(file.name).with_stem(file.stem + "_flip"))
            # 分为原图 和 已经被翻转的图两种情况
            if save_file_path.exists() or file.stem.endswith("_flip"):
                continue

            img = Image.open(file)
            result = ImageOps.flip(ImageOps.mirror(img))
            result.save(save_file_path)
        except Exception as e:
            print("Transform image failed:", e)

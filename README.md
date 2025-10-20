## 数据集处理流程

### 0. 扩充仅有单图的数据集

1. 获取单图训练素材后，用即梦清除背景并进行其他处理（比如生成人物的全身照）
2. 放大图片，下载后清除水印——由于图片已经是白色背景，所以可以用 mspaint 简单的涂抹掉水印
3. 使用 nanobanana 批量处理

### 0.1 半自动处理数据集

`uv run main.py <仅包含图片文件夹>`

### 1. 读取并使用 waifuc 处理

核心功能：图片三分并去重，其它 Action 请参考 waifuc 文档

```python
(LocalSource(source)).attach(
    ModeConvertAction("RGB", "white"),
    ThreeStageSplitAction(),
    CCIPAction(min_val_count=15),
    FilterSimilarAction(),
    FileOrderAction(),
    # TaggingAction(),
    FileExtAction(ext=".png"),
).export(TextualInversionExporter(dest))
```

### 2. 使用 cl-tagger 批量打标
```python
tag_cleaner = TagCleaner()

for image_path in Path(dest).glob("*.png"):
    filename = Path(image_path).with_suffix("")

    tags = process_image_and_save_tags(
        image_path=str(image_path),
        gen_threshold=0.60,
    )

    tag_cleaner.add_tags(filename=filename, tags=tags)
```

### 3. 手动清除 tags （仅适用于角色类的数据集）
``` python
for file, tags in tag_cleaner.get_cleaned_tags(round(total_images * 0.3)):
    file.with_suffix(".txt").write_text(', '.join(tags))
```
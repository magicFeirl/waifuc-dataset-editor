import os
import shutil

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif'}
FOLDER_PREFIX = 'animestyle'

folder = input("请输入文件夹地址: ").strip().strip('"')
start = int(input("请输入起始数字: ").strip())

# suffix = input("请输入文件夹后缀: ").strip().strip('"')
# if suffix:
#     suffix = '_' + suffix

images = sorted([
    f for f in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
])

if not images:
    print("未找到图片文件")
else:
    for i, filename in enumerate(images):
        dir_name = f'{FOLDER_PREFIX}{str(start + i)}{suffix}'
        dir_path = os.path.join(folder, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        shutil.move(os.path.join(folder, filename), os.path.join(dir_path, filename))
        print(f"{filename} -> {dir_name}/")
    print(f"完成，共处理 {len(images)} 张图片")

    print('请自行处理文件夹后缀')
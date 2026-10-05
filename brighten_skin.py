import cv2
import numpy as np

def optimize_skin_tone(img_path, output_path):
    # 1. 读取并转换空间
    img = cv2.imread(img_path)
    if img is None:
        return

    # 转换到 HSV 用于精准选区，转换到 LAB 用于高质量提亮
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. 精准定义二次元肤色选区 (针对 Anima 样图优化)
    # H(色调): 0-25 覆盖红橙黄; S(饱和度): 20-150 避开纯白背景
    lower_skin = np.array([0, 0, 30], dtype=np.uint8)
    upper_skin = np.array([200, 100, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 3. 关键：软化选区边缘，防止出现“贴纸感”
    skin_mask_blur = cv2.GaussianBlur(skin_mask, (15, 15), 0)
    skin_mask_float = skin_mask_blur.astype(float) / 255.0

    # 4. 在 LAB 空间提亮 L 通道
    # 提亮强度：1.2 代表提升 20%，你可以根据需求调整
    l_bright = np.clip(l.astype(float) * 1.5 + 10, 0, 255).astype(np.uint8)

    # 融合提亮后的 L 通道
    l_final = (l_bright * skin_mask_float + l * (1 - skin_mask_float)).astype(np.uint8)

    # 5. 合并并转回 BGR
    lab_final = cv2.merge((l_final, a, b))
    result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)

    # 6. 饱和度微调：防止变橙
    h, s, v = cv2.split(cv2.cvtColor(result, cv2.COLOR_BGR2HSV))
    # 提亮皮肤的地方，饱和度稍微降低 10%，看起来更清爽
    s_new = np.where(skin_mask > 0, np.clip(s.astype(float) * 0.9, 0, 255), s).astype(
        np.uint8
    )
    result = cv2.cvtColor(cv2.merge((h, s_new, v)), cv2.COLOR_HSV2BGR)

    cv2.imwrite(output_path, result)


optimize_skin_tone(r"E:\dataset\waifuc\output\ace_taffy_waifuc_anima\8.jpg", 'test.webp')

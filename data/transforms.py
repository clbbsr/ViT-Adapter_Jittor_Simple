import random
import numpy as np
from PIL import Image


class SimpleTrainTransform:
    """简化版训练数据增强"""

    def __init__(self, size=384):
        self.size = size

    def __call__(self, img, mask):
        # 1. 缩放到固定大小 (简化：直接resize)
        img = Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)

        # 2. 随机水平翻转
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 3. 随机亮度调整 (简化版)
        if random.random() > 0.5:
            img = np.array(img)
            brightness_factor = random.uniform(0.8, 1.2)
            img = img * brightness_factor
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        else:
            img = np.array(img)
            img = Image.fromarray(img)

        return np.array(img), np.array(mask)


class SimpleValTransform:
    """简化版验证变换"""

    def __init__(self, size=384):
        self.size = size

    def __call__(self, img, mask):
        # 只做resize
        img = Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)

        return np.array(img), np.array(mask)
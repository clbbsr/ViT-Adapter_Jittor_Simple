import random
import numpy as np
from PIL import Image


class SimpleTrainTransform:
    """简化版训练数据增强"""

    def __init__(self, size=384):
        self.size = size

    def __call__(self, img, mask):
        # 1. 确保mask是正确的数据类型
        if isinstance(mask, np.ndarray):
            if mask.dtype == np.int64:
                mask = mask.astype(np.uint8)

        # 2. 缩放到固定大小
        img = Image.fromarray(img.astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)

        # 3. 随机水平翻转
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 4. 随机亮度调整
        if random.random() > 0.5:
            img = np.array(img).astype(np.float32)
            brightness_factor = random.uniform(0.8, 1.2)
            img = img * brightness_factor
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)

        return np.array(img), np.array(mask, dtype=np.int64) 


class SimpleValTransform:
    """简化版验证变换"""

    def __init__(self, size=384):
        self.size = size

    def __call__(self, img, mask):
        # 转换数据类型
        if isinstance(mask, np.ndarray) and mask.dtype == np.int64:
            mask = mask.astype(np.uint8)

        # 只做resize
        img = Image.fromarray(img.astype(np.uint8)).resize((self.size, self.size), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST)

        return np.array(img), np.array(mask, dtype=np.int64)
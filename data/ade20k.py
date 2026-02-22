import os
import numpy as np
import jittor as jt
from PIL import Image


class ADE20KDataset(jt.dataset.Dataset):
    """ADE20K数据集"""

    def __init__(self,
                 root='data/ade/ADEChallengeData2016',
                 split='training',
                 transform=None,
                 reduce_zero_label=True,
                 ignore_index=255):

        super().__init__()

        self.root = root
        self.split = split
        self.transform = transform
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

        # 图像和标注路径
        if split == 'training':
            self.img_dir = os.path.join(root, 'images', 'training')
            self.ann_dir = os.path.join(root, 'annotations', 'training')
        else:
            self.img_dir = os.path.join(root, 'images', 'validation')
            self.ann_dir = os.path.join(root, 'annotations', 'validation')

        # 获取所有图像文件
        self.img_infos = []
        valid_extensions = ('.jpg', '.jpeg', '.png')

        for img_file in sorted(os.listdir(self.img_dir)):
            if img_file.lower().endswith(valid_extensions):
                img_name = os.path.splitext(img_file)[0]
                ann_file = img_name + '.png'

                # 检查标注文件是否存在
                if os.path.exists(os.path.join(self.ann_dir, ann_file)):
                    self.img_infos.append({
                        'img': img_file,
                        'ann': ann_file
                    })

        print(f"Loaded {len(self.img_infos)} images from {split} split")

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        img_info = self.img_infos[idx]

        # 加载图像
        img_path = os.path.join(self.img_dir, img_info['img'])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        # 加载标注
        ann_path = os.path.join(self.ann_dir, img_info['ann'])
        ann = Image.open(ann_path)
        ann = np.array(ann, dtype=np.int64)

        # 减少零标签
        if self.reduce_zero_label:
            ann = ann - 1
            ann[ann == -1] = self.ignore_index

        # 应用数据增强
        if self.transform:
            img, ann = self.transform(img, ann)

        # 归一化
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img = img.astype(np.float32)
        img = (img - mean) / std

        # 转换为CHW格式
        img = img.transpose(2, 0, 1)

        # 转换为Jittor数组
        img = jt.array(img, dtype=jt.float32)
        ann = jt.array(ann, dtype=jt.int32)

        return img, ann
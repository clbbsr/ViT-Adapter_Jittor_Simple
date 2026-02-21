import os
import sys
import argparse
import numpy as np
from PIL import Image
import jittor as jt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_adapter_mini import build_vit_adapter_mini
from models.semantic_fpn import SemanticFPN


def parse_args():
    parser = argparse.ArgumentParser(description='测试Semantic FPN with ViT-Adapter')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--image', type=str, required=True, help='测试图片路径')
    parser.add_argument('--image-size', type=int, default=384, help='输入图像尺寸')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU推理')
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    if not args.cpu:
        jt.flags.use_cuda = 1
        print("✓ 使用GPU推理")
    else:
        jt.flags.use_cuda = 0
        print("✓ 使用CPU推理")

    # 构建模型
    print("正在加载模型...")
    backbone = build_vit_adapter_mini()
    model = SemanticFPN(
        backbone=backbone,
        in_channels=[192, 192, 192, 192],
        num_classes=150
    )

    # 加载权重
    model.load(args.checkpoint)
    model.eval()
    print(f"✓ 模型加载成功: {args.checkpoint}")

    # 加载并预处理图片
    print(f"加载图片: {args.image}")
    img = Image.open(args.image).convert('RGB')
    original_size = img.size

    # 调整大小
    if img.size != (args.image_size, args.image_size):
        img = img.resize((args.image_size, args.image_size), Image.BILINEAR)

    # 归一化 (ImageNet统计)
    img = np.array(img, dtype=np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean) / std

    # 转换为CHW格式并增加batch维度
    img = img.transpose(2, 0, 1)[None, :, :, :]
    img = jt.array(img, dtype=jt.float32)

    # 推理
    print("正在进行推理...")
    with jt.no_grad():
        output = model(img)  # [1, 150, H, W]

    # 获取预测结果
    pred = output[0].argmax(0).numpy()  # [H, W]

    # 保存结果
    output_path = 'prediction.png'
    pred_img = Image.fromarray(pred.astype(np.uint8))
    pred_img.save(output_path)
    print(f"✓ 预测结果已保存: {output_path}")

    # 输出信息
    print(f"\n预测完成!")
    print(f"- 输入尺寸: {original_size}")
    print(f"- 输出形状: {pred.shape}")
    print(f"- 预测类别数: {len(np.unique(pred))}")

    # 显示类别分布
    unique, counts = np.unique(pred, return_counts=True)
    print("\n类别统计:")
    for cls, count in zip(unique[:5], counts[:5]):  # 只显示前5个
        print(f"  类别 {cls}: {count} 像素")
    if len(unique) > 5:
        print(f"  ... 等 {len(unique)} 个类别")


if __name__ == '__main__':
    main()
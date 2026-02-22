import argparse
import os
import sys
import time

import jittor as jt
from jittor import nn

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_adapter_mini import build_vit_adapter_mini
from models.semantic_fpn import SemanticFPN
from data.ade20k import ADE20KDataset
from data.transforms import SimpleTrainTransform
from utils.load_deit import load_deit_weights


if hasattr(jt, 'flags'):
    # Jittor旧版本
    HAS_AMP_SCOPE = hasattr(jt, 'amp_scope')
    # Jittor新版本
    HAS_AUTOCAST = hasattr(jt, 'autocast') or (hasattr(jt, 'amp') and hasattr(jt.amp, 'autocast'))
else:
    HAS_AMP_SCOPE = False
    HAS_AUTOCAST = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train Semantic FPN with ViT-Adapter')
    parser.add_argument('--data-root', type=str, default='data/ade/ADEChallengeData2016')
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--total-iters', type=int, default=80000)
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--work-dir', type=str, default='work_dirs/simple_run')
    parser.add_argument('--pretrained', type=str, default='pretrained/deit_tiny_patch16_224.pkl')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


class AverageMeter:
    """计算和存储平均值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train():
    args = parse_args()

    # 创建工作目录
    os.makedirs(args.work_dir, exist_ok=True)

    # 设置设备
    jt.flags.use_cuda = 1
    if args.fp16:
        if hasattr(jt, 'flags'):
            jt.flags.amp_level = 2
            print("✓ 启用混合精度训练 (通过 jt.flags.amp_level)")
        elif HAS_AMP_SCOPE:
            print("✓ 启用混合精度训练 (支持 amp_scope)")
        elif HAS_AUTOCAST:
            print("✓ 启用混合精度训练 (支持 autocast)")
        else:
            print("⚠️ 当前Jittor版本不支持混合精度，将使用FP32训练")
            args.fp16 = False
    else:
        if hasattr(jt, 'flags'):
            jt.flags.amp_level = 0

    print("=" * 50)
    print("ViT-Adapter 超快速训练")
    print("=" * 50)
    print(f"图像尺寸: {args.image_size}×{args.image_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"总迭代数: {args.total_iters}")
    print(f"工作目录: {args.work_dir}")
    print("=" * 50)

    # 1. 构建模型
    print("\n[1/5] 构建模型...")
    backbone = build_vit_adapter_mini()
    model = SemanticFPN(
        backbone=backbone,
        in_channels=[192, 192, 192, 192],
        num_classes=150
    )

    # 加载预训练权重
    if os.path.exists(args.pretrained):
        print(f"加载预训练权重: {args.pretrained}")
        model = load_deit_weights(model, args.pretrained)
    else:
        print(f"警告: 预训练权重不存在 {args.pretrained}")

    # 恢复训练
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        print(f"恢复训练: {args.resume}")
        model.load(args.resume)
        start_iter = int(args.resume.split('_')[-1].split('.')[0])

    # 2. 数据集
    print("\n[2/5] 加载数据集...")
    train_dataset = ADE20KDataset(
        root=args.data_root,
        split='training',
        transform=SimpleTrainTransform(args.image_size)
    )

    train_loader = jt.dataset.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    print(f"训练集大小: {len(train_dataset)} 张图片")

    # 3. 优化器
    print("\n[3/5] 配置优化器...")
    optimizer = jt.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # 4. 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 5. 训练循环
    print("\n[4/5] 开始训练...")
    model.train()
    train_iter = iter(train_loader)

    # 统计
    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    best_loss = float('inf')

    for iter_idx in range(start_iter, args.total_iters):
        start_time = time.time()

        # 获取数据
        try:
            images, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, targets = next(train_iter)

        # 学习率预热
        if iter_idx < args.warmup_iters:
            lr_scale = min(1., (iter_idx + 1) / args.warmup_iters)
            optimizer.lr = args.lr * lr_scale

        # 前向传播 - 根据Jittor版本选择不同方式
        if args.fp16:
            if HAS_AUTOCAST:
                # Jittor新版本
                with jt.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            elif HAS_AMP_SCOPE:
                # Jittor旧版本
                with jt.amp_scope(level=2):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                # 已经通过flags设置了amp_level
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        # 反向传播
        optimizer.step(loss)

        # 更新统计
        loss_meter.update(loss.data[0])
        time_meter.update(time.time() - start_time)

        # 日志
        if (iter_idx + 1) % 100 == 0:
            remaining_time = time_meter.avg * (args.total_iters - iter_idx - 1) / 3600
            print(f"Iter {iter_idx + 1:6d}/{args.total_iters} | "
                  f"Loss: {loss_meter.avg:.4f} | "
                  f"Time: {time_meter.avg * 1000:.1f}ms | "
                  f"Remaining: {remaining_time:.1f}h | "
                  f"LR: {optimizer.lr:.6f}")

            loss_meter.reset()

        # 保存模型
        if (iter_idx + 1) % 5000 == 0:
            save_path = os.path.join(args.work_dir, f'iter_{iter_idx + 1}.pkl')
            model.save(save_path)
            print(f"\n✓ 模型已保存: {save_path}")

            # 保存最佳模型
            if loss_meter.avg < best_loss:
                best_loss = loss_meter.avg
                best_path = os.path.join(args.work_dir, 'best.pkl')
                model.save(best_path)
                with open(os.path.join(args.work_dir, 'best_loss.txt'), 'w') as f:
                    f.write(str(best_loss))

    # 6. 保存最终模型
    print("\n[5/5] 训练完成，保存最终模型...")
    final_path = os.path.join(args.work_dir, 'final.pkl')
    model.save(final_path)
    print(f"✓ 最终模型已保存: {final_path}")

    # 保存训练配置
    with open(os.path.join(args.work_dir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    print("\n" + "=" * 50)
    print("训练成功完成！")
    print("=" * 50)


if __name__ == '__main__':
    train()
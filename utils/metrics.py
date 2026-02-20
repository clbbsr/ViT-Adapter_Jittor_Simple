import numpy as np


def compute_miou(pred, target, num_classes=150, ignore_index=255):
    """计算mIoU

    Args:
        pred: 预测标签 [H, W] 或 [B, H, W]
        target: 真实标签 [H, W] 或 [B, H, W]
        num_classes: 类别数
        ignore_index: 忽略的标签值

    Returns:
        miou: 平均IoU
        ious_per_class: 每个类别的IoU
    """
    if pred.ndim == 3:
        # 批处理
        batch_size = pred.shape[0]
        total_ious = []

        for i in range(batch_size):
            ious = _compute_miou_single(pred[i], target[i], num_classes, ignore_index)
            total_ious.append(ious)

        avg_ious = np.mean(total_ious, axis=0)
        return np.mean(avg_ious), avg_ious
    else:
        ious = _compute_miou_single(pred, target, num_classes, ignore_index)
        return np.mean(ious), ious


def _compute_miou_single(pred, target, num_classes, ignore_index):
    """计算单个样本的IoU"""
    ious = []

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        # 忽略 ignore_index
        valid_mask = (target != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()

        if union > 0:
            ious.append(intersection / union)
        elif target_mask.sum() > 0:
            # 目标存在但预测为0，IoU为0
            ious.append(0.0)
        # 如果目标和预测都不存在，跳过该类

    return np.array(ious)


def compute_accuracy(pred, target, ignore_index=255):
    """计算像素准确率"""
    valid = target != ignore_index
    correct = (pred == target) & valid
    accuracy = correct.sum() / valid.sum()
    return accuracy
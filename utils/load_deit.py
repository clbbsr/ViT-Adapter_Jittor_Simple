import jittor as jt
import pickle


def load_deit_weights(model, checkpoint_path):
    """加载DeiT预训练权重

    Args:
        model: Jittor模型
        checkpoint_path: PyTorch权重路径 (.pth 或 .pkl)

    Returns:
        加载权重后的模型
    """
    print(f"Loading DeiT weights from {checkpoint_path}")

    # 加载权重文件
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    # 提取state_dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 处理键名
    new_state_dict = {}
    model_state_dict = model.state_dict()

    for k, v in state_dict.items():
        # 移除不必要的前缀
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('backbone.'):
            k = k[9:]

        # 只加载backbone相关的权重
        if k in model_state_dict:
            # 检查形状是否匹配
            if model_state_dict[k].shape == v.shape:
                new_state_dict[k] = v
                print(f"  ✓ {k} -> {v.shape}")
            else:
                print(f"  ✗ {k} shape mismatch: {model_state_dict[k].shape} vs {v.shape}")

    # 加载权重
    for k, v in new_state_dict.items():
        if isinstance(v, jt.Var):
            model_state_dict[k].update(v)
        else:
            # 转换为Jittor数组
            model_state_dict[k].update(jt.array(v))

    print(f"Loaded {len(new_state_dict)}/{len(model_state_dict)} weights")

    return model
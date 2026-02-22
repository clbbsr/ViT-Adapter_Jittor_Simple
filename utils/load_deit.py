import jittor as jt
import torch


def load_deit_weights(model, checkpoint_path):
    """加载DeiT预训练权重"""
    print(f"Loading DeiT weights from {checkpoint_path}")

    # 使用torch加载
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ 使用torch加载成功")
    except Exception as e:
        print(f"torch加载失败: {e}")
        return model

    # 提取state_dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("✓ 从'model'键提取")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✓ 从'state_dict'键提取")
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']
            print("✓ 从'module'键提取")
        else:
            state_dict = checkpoint
            print("✓ 直接使用state_dict")
    else:
        state_dict = checkpoint

    # 转换为numpy并加载
    model_params = dict(model.named_parameters())
    loaded_count = 0

    for name, param in model_params.items():
        # 尝试不同的键名
        if name in state_dict:
            weights = state_dict[name]
        elif name.startswith('backbone.') and name[9:] in state_dict:
            weights = state_dict[name[9:]]
            name = name[9:]  # 更新显示名称
        else:
            continue

        # 转换为numpy
        if isinstance(weights, torch.Tensor):
            weights = weights.cpu().numpy()

        # 检查形状并加载
        if param.shape == weights.shape:
            param.update(jt.array(weights))
            loaded_count += 1
            if loaded_count <= 5:  # 只显示前5个
                print(f"  ✓ {name} -> {weights.shape}")
        else:
            print(f"  ✗ {name} shape mismatch: {param.shape} vs {weights.shape}")

    print(f"✓ 成功加载 {loaded_count}/{len(model_params)} 个权重")
    return model
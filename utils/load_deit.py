import jittor as jt
import pickle
import numpy as np


def load_deit_weights(model, checkpoint_path):
    """加载DeiT预训练权重"""
    print(f"Loading DeiT weights from {checkpoint_path}")

    try:
        # 使用pickle直接加载
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print("✓ 使用pickle加载成功")
    except Exception as e:
        print(f"加载失败: {e}")
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
        print("✗ 无法识别的权重格式")
        return model

    # 获取模型参数
    model_params = dict(model.named_parameters())
    loaded_count = 0
    skipped_count = 0

    print("\n加载权重:")
    for name, param in model_params.items():
        # 尝试匹配权重名称
        if name in state_dict:
            weights = state_dict[name]
        elif name.startswith('backbone.') and name[9:] in state_dict:
            weights = state_dict[name[9:]]
            name_show = name[9:]
        else:
            skipped_count += 1
            continue

        # 转换为numpy数组
        if hasattr(weights, 'numpy'):  # torch tensor或jittor var
            weights = weights.numpy()
        elif isinstance(weights, np.ndarray):
            pass
        else:
            try:
                weights = np.array(weights)
            except:
                skipped_count += 1
                continue

        # 检查形状并加载
        if param.shape == weights.shape:
            param.update(jt.array(weights))
            loaded_count += 1
            if loaded_count <= 5:  # 只显示前5个
                print(f"  ✓ {name} -> {weights.shape}")
        else:
            print(f"  ✗ {name} shape mismatch: {param.shape} vs {weights.shape}")
            skipped_count += 1

    print(f"\n✓ 成功加载 {loaded_count}/{len(model_params)} 个权重 (跳过 {skipped_count} 个)")
    return model
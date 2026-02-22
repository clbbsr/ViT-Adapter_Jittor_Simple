import jittor as jt
import pickle
import numpy as np


def load_deit_weights(model, checkpoint_path):
    """加载DeiT预训练权重（修正版）"""
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

    # 获取模型参数
    model_params = dict(model.named_parameters())

    loaded_count = 0
    for name, param in model_params.items():
        # 构建对应的权重键名
        key = name
        if key in state_dict:
            # 转换为numpy再转换为jittor
            if isinstance(state_dict[key], jt.Var):
                weights = state_dict[key].numpy()
            elif isinstance(state_dict[key], np.ndarray):
                weights = state_dict[key]
            else:
                weights = state_dict[key].cpu().numpy() if hasattr(state_dict[key], 'cpu') else state_dict[key]

            # 检查形状
            if param.shape == weights.shape:
                param.update(jt.array(weights))
                loaded_count += 1
                print(f"  ✓ {name} -> {weights.shape}")
            else:
                print(f"  ✗ {name} shape mismatch: {param.shape} vs {weights.shape}")

    print(f"Loaded {loaded_count}/{len(model_params)} weights")
    return model
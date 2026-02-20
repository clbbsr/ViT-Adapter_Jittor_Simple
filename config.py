class Config:
    """统一配置类"""

    # ==================== 模型配置 ====================
    MODEL_TYPE = 'vit_adapter_mini'  # 模型类型
    EMBED_DIM = 192  # 嵌入维度
    NUM_HEADS = 3  # 注意力头数
    DEPTH = 12  # Transformer层数
    MLP_RATIO = 4  # MLP扩展比例

    # ==================== 数据配置 ====================
    DATA_ROOT = 'data/ade/ADEChallengeData2016'  # 数据集根目录
    IMAGE_SIZE = 384  # 输入图像尺寸
    CROP_SIZE = 384  # 裁剪尺寸
    NUM_CLASSES = 150  # 类别数
    IGNORE_INDEX = 255  # 忽略的标签值

    # 归一化参数 (ImageNet统计)
    MEAN = [123.675, 116.28, 103.53]
    STD = [58.395, 57.12, 57.375]

    # ==================== 训练配置 ====================
    BATCH_SIZE = 8  # 批次大小
    NUM_WORKERS = 0  # 数据加载线程数 (Jittor设为0)
    TOTAL_ITERS = 80000  # 总迭代次数
    LR = 1e-4  # 学习率
    WEIGHT_DECAY = 0.01  # 权重衰减
    WARMUP_ITERS = 1000  # 预热迭代数
    LOG_INTERVAL = 100  # 日志间隔
    SAVE_INTERVAL = 5000  # 保存间隔

    # ==================== 加速配置 ====================
    FP16 = True  # 混合精度训练
    SIMPLE_AUG = True  # 简单数据增强

    # ==================== 路径配置 ====================
    PRETRAINED_PATH = 'pretrained/deit_tiny_patch16_224.pkl'  # 预训练权重
    WORK_DIR = 'work_dirs/vit_adapter_simple'  # 工作目录

    @classmethod
    def to_dict(cls):
        """转换为字典"""
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('_') and not callable(v)}


# 创建配置实例
config = Config()
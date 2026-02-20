```bash
#!/bin/bash
# run.sh - 一键训练脚本

set -e  # 遇到错误立即退出

echo "========================================="
echo "ViT-Adapter Jittor 超快速训练脚本"
echo "========================================="

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查GPU可用性
echo -n "检查GPU状态... "
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        echo -e "${GREEN}可用 (找到 $GPU_COUNT 张GPU)${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}警告: 未找到GPU，将使用CPU训练${NC}"
    fi
else
    echo -e "${YELLOW}警告: nvidia-smi 未找到，将使用CPU训练${NC}"
fi

# 检查Python环境
echo -n "检查Python环境... "
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version)
    echo -e "${GREEN}$PY_VERSION${NC}"
else
    echo -e "${RED}错误: 未找到python3${NC}"
    exit 1
fi

# 检查依赖
echo "检查Python依赖..."
python3 -c "import jittor" 2>/dev/null || { echo -e "${RED}错误: jittor 未安装${NC}"; exit 1; }
python3 -c "import numpy" 2>/dev/null || { echo -e "${RED}错误: numpy 未安装${NC}"; exit 1; }
python3 -c "import PIL" 2>/dev/null || { echo -e "${RED}错误: Pillow 未安装${NC}"; exit 1; }
echo -e "${GREEN}所有依赖已安装${NC}"

# 检查数据集
echo -n "检查ADE20K数据集... "
if [ -d "data/ade/ADEChallengeData2016" ]; then
    IMG_COUNT=$(ls -1 data/ade/ADEChallengeData2016/images/training | wc -l)
    echo -e "${GREEN}找到 (训练集: $IMG_COUNT 张图片)${NC}"
else
    echo -e "${YELLOW}未找到，将自动下载${NC}"
    mkdir -p data/ade
    cd data/ade
    echo "下载ADE20K数据集 (约3GB)..."
    wget -q --show-progress http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
    echo "解压中..."
    unzip -q ADEChallengeData2016.zip
    rm ADEChallengeData2016.zip
    cd ../..
    echo -e "${GREEN}数据集下载完成${NC}"
fi

# 检查预训练权重
echo -n "检查预训练权重... "
if [ -f "pretrained/deit_tiny_patch16_224.pkl" ]; then
    echo -e "${GREEN}已找到${NC}"
else
    echo -e "${YELLOW}未找到，将自动下载${NC}"
    mkdir -p pretrained
    cd pretrained
    echo "下载DeiT-Tiny权重..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth -O deit_tiny_patch16_224.pth
    echo "转换为Jittor格式..."
    python3 -c "
import torch
import pickle
data = torch.load('deit_tiny_patch16_224.pth', map_location='cpu')
with open('deit_tiny_patch16_224.pkl', 'wb') as f:
    pickle.dump(data, f)
print('转换完成')
"
    cd ..
    echo -e "${GREEN}权重准备完成${NC}"
fi

# 创建日志目录
mkdir -p logs

# 设置训练参数
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="work_dirs/train_${TIMESTAMP}"
LOG_FILE="logs/train_${TIMESTAMP}.log"

echo "========================================="
echo "训练配置:"
echo "- 工作目录: $WORK_DIR"
echo "- 日志文件: $LOG_FILE"
echo "- 图像尺寸: 384×384"
echo "- Batch size: 8"
echo "- 学习率: 1e-4"
echo "- 迭代次数: 80000"
echo "- 混合精度: 开启"
echo "========================================="

# 开始训练
echo -e "${GREEN}开始训练...${NC}"
python3 -m utils.train \
    --data-root data/ade/ADEChallengeData2016 \
    --image-size 384 \
    --batch-size 8 \
    --lr 1e-4 \
    --total-iters 80000 \
    --fp16 \
    --work-dir "$WORK_DIR" 2>&1 | tee "$LOG_FILE"

# 检查训练结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}训练成功完成！${NC}"
    echo "模型保存在: $WORK_DIR"

    # 显示最佳结果
    if [ -f "$WORK_DIR/best_miou.txt" ]; then
        BEST_MIOU=$(cat "$WORK_DIR/best_miou.txt")
        echo -e "${GREEN}最佳 mIoU: $BEST_MIOU${NC}"
    fi
else
    echo -e "${RED}训练失败，请检查日志: $LOG_FILE${NC}"
    exit 1
fi

echo "========================================="
echo "训练完成！"
echo "可使用以下命令测试模型："
echo "python -m utils.test --checkpoint $WORK_DIR/final.pkl"
echo "========================================="
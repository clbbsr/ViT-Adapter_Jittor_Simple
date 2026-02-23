本项目是对论文 Vision Transformer Adapter for Dense Predictions 的 Jittor 框架复现实现。作为一次学习实践，我在资源有限的情况下，使用 ViT-Tiny + Semantic FPN 在 ADE20K 数据集上完成了语义分割任务的复现验证。

复现目标
本次复现是一个学习项目，主要目标是：
深入理解 ViT-Adapter 的设计思想和 Adapter 范式
掌握 Jittor 框架的使用和深度学习实践
在资源有限的情况下验证代码正确性
完整经历从论文到代码的实现过程

# 克隆仓库
git clone https://github.com/yourusername/ViT-Adapter_Jittor_Simple.git
cd ViT-Adapter_Jittor_Simple

# 验证 Jittor 安装
python -c "import jittor as jt; print(f'Jittor版本: {jt.__version__}')"

# 创建数据目录
mkdir -p data/ade

# 下载 ADE20K 数据集（约3GB，推荐使用 aria2 加速）
cd data/ade
aria2c -x 16 -s 16 http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip -q ADEChallengeData2016.zip
cd ../..

# 运行100次迭代验证代码正确性
python -m utils.train \
    --data-root data/ade/ADEChallengeData2016 \
    --image-size 224 \
    --batch-size 2 \
    --total-iters 100 \
    --work-dir work_dirs/quick_test
    
# 快速训练配置
python -m utils.train \
    --data-root data/ade/ADEChallengeData2016 \
    --image-size 224 \
    --batch-size 8 \
    --lr 1e-4 \
    --total-iters 10000 \
    --fp16 \
    --work-dir work_dirs/vit_adapter_run
    
# 单张图片测试
python -m utils.test \
    --checkpoint work_dirs/vit_adapter_run/final.pkl \
    --image test.jpg
    
实验结果
训练曲线
经过 100 次迭代验证，损失函数从 5.5 降至 5.04，呈平稳下降趋势，证明代码逻辑正确。

分割效果
使用训练 10k 次的模型在 ADE20K 验证集上测试：
预测类别数：42 类
推理时间：1.67 秒/张（224×224）
模型大小：约 30MB
虽然精度有限（mIoU 约 30%），但模型确实学到了语义信息，验证了方法的可行性。


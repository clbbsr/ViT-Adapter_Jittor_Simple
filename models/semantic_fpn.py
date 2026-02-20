import jittor.nn as nn


class ConvModule(nn.Module):
    """卷积模块：Conv + BN + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FPNHead(nn.Module):
    """Semantic FPN Head"""

    def __init__(self, in_channels=[192, 192, 192, 192],
                 channels=256, num_classes=150, dropout_ratio=0.1):
        super().__init__()

        # 四个层级的处理头
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(in_channels)):
            lateral_conv = ConvModule(in_channels[i], channels, 1)
            fpn_conv = ConvModule(channels, channels, 3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # 最终分类层
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def execute(self, inputs):
        # 输入: [f1, f2, f3, f4] 对应分辨率 [1/4, 1/8, 1/16, 1/32]

        # 侧向连接
        laterals = []
        for i, (x, lateral) in enumerate(zip(inputs, self.lateral_convs)):
            laterals.append(lateral(x))

        # 自顶向下融合
        fpn = []
        for i in range(len(laterals) - 1, 0, -1):
            if laterals[i].shape[2:] != laterals[i - 1].shape[2:]:
                laterals[i - 1] = laterals[i - 1] + nn.interpolate(
                    laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear'
                )

        # 应用FPN卷积
        for i, (x, fpn_conv) in enumerate(zip(laterals, self.fpn_convs)):
            x = fpn_conv(x)
            fpn.append(x)

        # 融合所有层级的特征
        x = fpn[0]
        for i in range(1, len(fpn)):
            if fpn[i].shape[2:] != x.shape[2:]:
                fpn[i] = nn.interpolate(fpn[i], size=x.shape[2:], mode='bilinear')
            x = x + fpn[i]

        x = self.dropout(x)
        x = self.conv_seg(x)

        return x


class SemanticFPN(nn.Module):
    """Semantic FPN 完整模型"""

    def __init__(self, backbone, in_channels=[192, 192, 192, 192],
                 num_classes=150, dropout_ratio=0.1):
        super().__init__()
        self.backbone = backbone
        self.decode_head = FPNHead(in_channels, num_classes=num_classes, dropout_ratio=dropout_ratio)

    def execute(self, x):
        # 主干网络提取特征
        feats = self.backbone(x)  # [f1, f2, f3, f4]

        # 解码
        out = self.decode_head(feats)

        # 上采样到原始大小
        if out.shape[2:] != x.shape[2:]:
            out = nn.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out

    def loss(self, images, targets, criterion):
        """计算损失"""
        outputs = self.execute(images)

        # 确保输出和targets大小一致
        if outputs.shape[2:] != targets.shape[1:]:
            outputs = nn.interpolate(outputs, size=targets.shape[1:], mode='bilinear')

        loss = criterion(outputs, targets)
        return loss
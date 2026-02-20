import jittor as jt
import jittor.nn as nn
import math


class QuickGELU(nn.Module):
    """快速GELU激活函数 (比标准GELU快约20%)"""

    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)


class PatchEmbed(nn.Module):
    """图像分块嵌入层"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 卷积实现分块
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def execute(self, x):
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        x = self.norm(x)
        return x, H, W


class SimplifiedAttention(nn.Module):
    """简化版多头注意力 (去除不必要的复杂操作)"""

    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x, H=None, W=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, dim, num_heads=3, mlp_ratio=4, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimplifiedAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.drop_path = drop_path

    def execute(self, x, H, W):
        # 残差连接
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialPriorModule(nn.Module):
    """简化版空间先验模块 (3层卷积)"""

    def __init__(self, in_chans=3, embed_dim=192):
        super().__init__()
        # 3层卷积快速提取空间特征
        self.conv1 = nn.Conv2d(in_chans, 64, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, embed_dim, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.act3 = nn.ReLU()

    def execute(self, x):
        x = self.act1(self.bn1(self.conv1(x)))  # 1/2
        x = self.act2(self.bn2(self.conv2(x)))  # 1/4
        x = self.act3(self.bn3(self.conv3(x)))  # 1/8

        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return x_flat, H, W


class ViTAdapterMini(nn.Module):
    """超简化版ViT-Adapter (约300行代码)"""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=192,
                 depth=12,
                 num_heads=3,
                 mlp_ratio=4,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        # 2. Position Embedding
        self.pos_embed = nn.Parameter(jt.zeros((1, num_patches, embed_dim)))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        # 3. Spatial Prior Module
        self.spm = SpatialPriorModule(in_chans, embed_dim)

        # 4. Transformer Blocks
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i]
            )
            for i in range(depth)
        ])

        # 5. 简化特征融合 (1x1卷积)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1)

        # 6. 输出归一化层
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        # 7. 上采样层
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)  # 1/8 -> 1/4
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)  # 1/16 -> 1/8
        self.pool = nn.MaxPool2d(2)  # 1/16 -> 1/32

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def execute(self, x):
        B, _, H, W = x.shape

        # 1. 空间特征提取
        spm_feat, H_spm, W_spm = self.spm(x)  # [B, H_spm*W_spm, C]

        # 2. ViT特征提取
        vit_feat, H_vit, W_vit = self.patch_embed(x)  # [B, H_vit*W_vit, C]
        vit_feat = vit_feat + self.pos_embed
        vit_feat = self.pos_drop(vit_feat)

        # 3. 通过Transformer块
        for blk in self.blocks:
            vit_feat = blk(vit_feat, H_vit, W_vit)

        # 4. 转换为2D特征图
        vit_feat_2d = vit_feat.transpose(1, 2).reshape(B, -1, H_vit, W_vit)
        spm_feat_2d = spm_feat.transpose(1, 2).reshape(B, -1, H_spm, W_spm)

        # 5. 调整spm特征到vit大小
        if H_spm != H_vit or W_spm != W_vit:
            spm_feat_2d = nn.interpolate(spm_feat_2d, size=(H_vit, W_vit), mode='bilinear')

        # 6. 特征融合
        fused = jt.concat([vit_feat_2d, spm_feat_2d], dim=1)
        fused = self.fusion(fused)  # [B, C, H_vit, W_vit]

        # 7. 生成多尺度特征
        f4 = fused  # 1/16
        f3 = self.up2(f4)  # 1/8
        f2 = self.up1(f3)  # 1/4
        f1 = nn.interpolate(f2, scale_factor=2, mode='bilinear')  # 1/2

        # 8. 归一化
        f1 = self.norm1(f1)
        f2 = self.norm2(f2)
        f3 = self.norm3(f3)
        f4 = self.norm4(f4)

        return [f1, f2, f3, f4]


def build_vit_adapter_mini(pretrained_path=None):
    """构建ViT-Adapter Mini模型"""
    model = ViTAdapterMini(
        img_size=224,
        embed_dim=192,
        depth=12,
        num_heads=3
    )

    # 可以在这里加载预训练权重
    if pretrained_path:
        try:
            model.load(pretrained_path)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except:
            print(f"Failed to load pretrained weights from {pretrained_path}")

    return model
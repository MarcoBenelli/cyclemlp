import torch
import torchvision
import timm


class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CycleFC(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
    ):
        super(CycleFC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, 1, 1))  # kernel size == 1

        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=5**(1/2))

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / fan_in**(1/2)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size,
            2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, \
            self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = \
                    (i + start_idx) % self.kernel_size[1] \
                    - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = \
                    (i + start_idx) % self.kernel_size[0] \
                    - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor[batch_size, in_channels, in_height, in_width]): x tensor
        """
        B, C, H, W = x.size()
        return torchvision.ops.deform_conv.deform_conv2d(
            x, self.offset.expand(B, -1, H, W), self.weight, self.bias)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ')'
        return s.format(**self.__dict__)


class CycleMLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_c = torch.nn.Linear(dim, dim, bias=False)

        self.sfc_h = CycleFC(dim, dim, (1, 3))
        self.sfc_w = CycleFC(dim, dim, (3, 1))

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0)\
            .unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)

        return x


class CycleBlock(torch.nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = CycleMLP(dim)

        self.norm2 = torch.nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedOverlapping(torch.nn.Module):
    """2D Image to Patch Embedding with overlapping"""

    def __init__(self, patch_size=16, stride=16,
                 padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding = (padding, padding)
        # remove image_size in model init to support dynamic image size

        self.proj = torch.nn.Conv2d(in_chans, embed_dim,
                                    kernel_size=patch_size,
                                    stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(torch.nn.Module):
    """Downsample transition stage"""

    def __init__(self, in_embed_dim, out_embed_dim):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_embed_dim, out_embed_dim,
                                    kernel_size=(3, 3),
                                    stride=(2, 2), padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, num_layers, mlp_ratio=3.):
    blocks = []

    for block_idx in range(num_layers):
        blocks.append(CycleBlock(dim, mlp_ratio=mlp_ratio))
    blocks = torch.nn.Sequential(*blocks)

    return blocks


class CycleNet(torch.nn.Module):
    """CycleMLP Network"""

    def __init__(self, layers, in_chans=3, num_classes=1000,
                 embed_dims=None, mlp_ratios=None):

        super().__init__()

        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4,
                                                 padding=2, in_chans=in_chans,
                                                 embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], layers[i],
                                 mlp_ratio=mlp_ratios[i])
            network.append(stage)
            if i >= len(layers) - 1:
                break
            network.append(Downsample(embed_dims[i], embed_dims[i+1]))

        self.network = torch.nn.ModuleList(network)

        # Classifier head
        self.norm = torch.nn.LayerNorm(embed_dims[-1])
        self.head = torch.nn.Linear(embed_dims[-1], num_classes)
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            timm.models.layers.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, CycleFC):
            timm.models.layers.trunc_normal_(m.weight, std=.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        outs = []
        for _, block in enumerate(self.network):
            x = block(x)

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)

        x = self.norm(x)
        cls_out = self.head(x.mean(1))
        return cls_out


def CycleMLP_B1(**kwargs):
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = CycleNet(layers, embed_dims=embed_dims,
                     mlp_ratios=mlp_ratios, **kwargs)
    return model

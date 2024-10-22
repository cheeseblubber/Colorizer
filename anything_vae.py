import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU()

        # Define conv_shortcut only when in_channels != out_channels
        self.use_conv_shortcut = in_channels != out_channels
        if self.use_conv_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        return x + h

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.group_norm(x)
        h = h.view(batch, channels, -1).permute(0, 2, 1)  # (batch, num_patches, channels)

        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)

        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (channels ** 0.5), dim=-1)
        attn_output = torch.bmm(attn_weights, v)

        attn_output = self.to_out(attn_output)
        attn_output = attn_output.permute(0, 2, 1).view(batch, channels, height, width)

        return x + attn_output

class Downsample2D(nn.Module):
    def __init__(self, channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample2D(nn.Module):
    def __init__(self, channels):
        super(Upsample2D, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2, downsample=True):
        super(DownEncoderBlock2D, self).__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, out_channels),
            ResnetBlock2D(out_channels, out_channels)
        ])
        self.downsamplers = nn.ModuleList()
        if downsample:
            self.downsamplers.append(Downsample2D(out_channels))

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x

class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=3, upsample=True):
        super(UpDecoderBlock2D, self).__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_res_blocks):
            res_in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    res_in_channels,
                    out_channels
                )
            )
        self.upsamplers = nn.ModuleList()
        if upsample:
            self.upsamplers.append(Upsample2D(out_channels))

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x

class UNetMidBlock2D(nn.Module):
    def __init__(self, channels):
        super(UNetMidBlock2D, self).__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels),
            ResnetBlock2D(channels, channels)
        ])
        self.attentions = nn.ModuleList([
            SelfAttention(channels)
        ])

    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([
            DownEncoderBlock2D(128, 128, num_res_blocks=2, downsample=True),
            DownEncoderBlock2D(128, 256, num_res_blocks=2, downsample=True),
            DownEncoderBlock2D(256, 512, num_res_blocks=2, downsample=True),
            DownEncoderBlock2D(512, 512, num_res_blocks=2, downsample=False),
        ])

        self.mid_block = UNetMidBlock2D(512)

        self.conv_norm_out = nn.GroupNorm(32, 512, eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(512, 8, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_in = nn.Conv2d(4, 512, kernel_size=3, padding=1)

        self.mid_block = UNetMidBlock2D(512)

        self.up_blocks = nn.ModuleList([
            UpDecoderBlock2D(512, 512, num_res_blocks=3, upsample=True),
            UpDecoderBlock2D(512, 512, num_res_blocks=3, upsample=True),
            UpDecoderBlock2D(512, 256, num_res_blocks=3, upsample=True),
            UpDecoderBlock2D(256, 128, num_res_blocks=3, upsample=False),
        ])

        self.conv_norm_out = nn.GroupNorm(32, 128, eps=1e-6, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x

class AutoencoderKL(nn.Module):
    def __init__(self):
        super(AutoencoderKL, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, z):
        z = self.post_quant_conv(z)
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg_pretrained = torchvision.models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg_pretrained.features.children())[:16])
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x_vgg = self.features(x)
        y_vgg = self.features(y)
        loss = self.criterion(x_vgg, y_vgg)
        return loss


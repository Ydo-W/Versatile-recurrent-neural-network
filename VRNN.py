import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare(centralized, normalized, x):
    rgb = 255.0
    if centralized:
        x = x - rgb / 2
    if normalized:
        x = x / rgb
    return x


def prepare_reverse(centralized, normalized, x):
    rgb = 255.0
    if normalized:
        x = x * rgb
    if centralized:
        x = x + rgb / 2
    return x


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)  # C = in_channels + growthRate
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate))  # 16->32->48->64
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):  # i = 0, 1, 2
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)  # (bs, 240, 64, 64)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 5, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 5, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class fusion_new(nn.Module):
    def __init__(self, ff=2, fb=2, n_feats = 16):
        super(fusion_new, self).__init__()
        self.time_attention = ChannelAttention(160)
        self.space_attention = SpatialAttention()
        self.center = ff  # 参考帧的位置，是2
        self.conv1 = conv1x1(160, 80)
        self.conv2 = conv1x1((5*n_feats)*(ff*2+1), (5*n_feats)*(ff*2+1))

    def forward(self, fm):
        self.nframes = len(fm)
        f_ref = fm[self.center]
        out = []
        for i in range(self.nframes):
            if i != self.center:
                map_cated = torch.cat([f_ref, fm[i]], dim = 1)
                map_cated = self.time_attention(map_cated) * map_cated
                map_cated = self.space_attention(map_cated) * map_cated
                map_cated = self.conv1(map_cated)
                out.append(map_cated)
        out.append(f_ref)
        out = self.conv2(torch.cat(out, dim=1))
        return out


class Reconstructor(nn.Module):
    def __init__(self, ff=2, fb=2, n_feats = 16):
        super(Reconstructor, self).__init__()
        self.num_ff = ff
        self.num_fb = fb
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = n_feats
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            RDB(in_channels=2*self.n_feats, growthRate=self.n_feats, num_layer=3),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3),
            conv5x5(self.n_feats, 3, stride=1)
        )

    def forward(self, x):
        return self.model(x)


# RNN cell
class RNNcell(nn.Module):
    def __init__(self, n_blocks=9, n_feats=16):
        super(RNNcell, self).__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = nn.Sequential(
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3),
            conv5x5(self.n_feats, 2 * self.n_feats, stride=2)
        )
        self.F_B2 = nn.Sequential(
            RDB(in_channels=2*self.n_feats, growthRate=2*self.n_feats, num_layer=3),
            conv5x5(2 * self.n_feats, 4 * self.n_feats, stride=2)
        )
        self.F_R = RDNet(in_channels=(1 + 4) * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks)  # in: 80
        self.F_h = nn.Sequential(
            conv3x3((1 + 4) * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3),
            conv3x3(self.n_feats, self.n_feats)
        )

    def forward(self, x, s_last):  # (bs, 3, 256, 256)
        out = self.F_B0(x)  # (bs, 16, 256, 256)
        out = self.F_B1(out)  # (bs, 32, 128, 128)
        out = self.F_B2(out)  # (bs, 64, 64, 64)
        out = torch.cat([out, s_last], dim=1)  # (bs, 80, 64, 64)
        out = self.F_R(out)  # (bs, 80, 64, 64)
        s = self.F_h(out)  # (bs, 16, 64, 64)
        return out, s


# VRNN
class Model(nn.Module):
    def __init__(self, ff=2, fb=2, n_blocks=9, n_feats = 16):
        super(Model, self).__init__()
        self.n_feats = n_feats
        self.num_ff = ff
        self.num_fb = fb
        self.ds_ratio = 4
        self.device = torch.device('cuda')
        self.cell = RNNcell(n_blocks)  # Forward RNN
        self.cell1 = RNNcell(n_blocks)  # Backward RNN
        self.recons = Reconstructor(ff, fb, n_feats)
        self.fusion = fusion_new(ff, fb, n_feats)
        self.y_Merge = conv1x1(10 * self.n_feats, 5 * self.n_feats)

    def forward(self, x):
        x = prepare(False, True, x)
        outputs, hs0, hs1, hs = [], [], [], []
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)  # s : batch_size, 16, 64, 64
        for i in range(frames):
            h, s = self.cell(x[:, i, :, :, :], s)
            hs0.append(h)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames - 1, -1, -1):
            h, s = self.cell1(x[:, i, :, :, :], s)
            hs1.append(h)
        for i in range(frames):
            s = torch.cat([hs1[frames - 1 - i], hs0[i]], dim=1)
            s = self.y_Merge(s)
            hs.append(s)
        for i in range(self.num_fb, frames - self.num_ff):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            out = out + x[:, i, :, :, :]
            outputs.append(out.unsqueeze(dim=1))
        return prepare_reverse(False, True, torch.cat(outputs, dim=1))



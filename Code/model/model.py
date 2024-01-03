import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from parameters import get_parameters
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

np.set_printoptions(suppress=True, threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

# 参数读取
parameters = get_parameters()
cuda = parameters['cuda']
device = parameters['device']
learning_rate = parameters['learning_rate']
early_stop_patients = parameters['early_stop_patients']
batch_size = parameters['batch_size']
epochs = parameters['epochs']
heat_width = parameters['heat_width']
heat_height = parameters['heat_height']
sea_surface_width = parameters['sea_surface_width']
sea_surface_height = parameters['sea_surface_height']
cha_width = parameters['cha_width']
cha_height = parameters['cha_height']
input_days = parameters['input_days']
output_days = parameters['output_days']
total_feature_count = parameters['total_feature_count']
patch_count = parameters['patch_count']
patch_heat_width = parameters['patch_heat_width']
patch_heat_height = parameters['patch_heat_height']
patch_sea_surface_width = parameters['patch_sea_surface_width']
patch_sea_surface_height = parameters['patch_sea_surface_height']
patch_cha_width = parameters['patch_cha_width']
patch_cha_height = parameters['patch_cha_height']
d_model = parameters['d_model']
n_heads = parameters['n_heads']
blocks = parameters['blocks']
conv_channel = parameters['conv_channel']


class Sea_Surface_FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_days, out_channels=conv_channel * 2, kernel_size=(3, 3), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(3, 3), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_days, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features


class Cha_FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_days, out_channels=conv_channel * 2, kernel_size=(3, 3), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(3, 3), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_days, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features


class Curr_FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_days * 3, out_channels=conv_channel * 2, kernel_size=(3, 3), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(3, 3), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_days, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features


class EarlyBird_FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_days, out_channels=conv_channel * 2, kernel_size=(3, 3), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(3, 3), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_days, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features


class Encoder_Heat_FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_days, out_channels=conv_channel * 2, kernel_size=(3, 3), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(3, 3), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=output_days, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()

        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # 输入输出维度一样[batch, channel, width, height]
        out = self.resblock(x)
        return out + x


class Hydrological_Feature_Construction(nn.Module):
    def __init__(self):
        super(Hydrological_Feature_Construction, self).__init__()
        self.sst_block = Sea_Surface_FeatureExtraction()
        self.sss_block = Sea_Surface_FeatureExtraction()
        self.ssh_block = Sea_Surface_FeatureExtraction()
        self.cha_block = Cha_FeatureExtraction()
        self.curr_block = Curr_FeatureExtraction()
        self.earlybird = EarlyBird_FeatureExtraction()
        self.heatmap = Encoder_Heat_FeatureExtraction()
        self.sst_projection = nn.Linear(sea_surface_width * sea_surface_height, 1)
        self.sss_projection = nn.Linear(sea_surface_width * sea_surface_height, 1)
        self.ssh_projection = nn.Linear(sea_surface_width * sea_surface_height, 1)
        self.cha_projection = nn.Linear(cha_width * cha_height, 1)
        self.cur_projection = nn.Linear(sea_surface_width * sea_surface_height, 1)
        self.sea_surface_split = Rearrange('b o (w p1) (h p2) -> b o (w h) (p1 p2)',
                                           p1=patch_sea_surface_width,
                                           p2=patch_sea_surface_height).to(device)
        self.heat_split = Rearrange('b o (w p1) (h p2) -> b o (w h) (p1 p2)',
                                    p1=patch_heat_width,
                                    p2=patch_heat_height).to(device)
        self.cha_split = Rearrange('b o (w p1) (h p2) -> b o (w h) (p1 p2)',
                                   p1=patch_cha_width,
                                   p2=patch_cha_height).to(device)

    def forward(self, heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha):
        # heatmap           =====> torch.Size([8, 14, 48, 48])
        # earlybird_heatmap =====> torch.Size([8, 14, 48, 48])
        # ssh               =====> torch.Size([8, 14, 24, 24])
        # sst               =====> torch.Size([8, 14, 24, 24])
        # sss               =====> torch.Size([8, 14, 24, 24])
        # curr              =====> torch.Size([8, 14, 3, 24, 24])
        # cha               =====> torch.Size([8, 14, 144, 144])

        sst_features = self.sst_block(sst)  # torch.Size([8, 7, 24, 24])
        sss_features = self.sss_block(sss)  # torch.Size([8, 7, 24, 24])
        ssh_features = self.ssh_block(ssh)  # torch.Size([8, 7, 24, 24])
        curr_features = self.curr_block(curr.view(curr.size(0), -1, curr.size(3), curr.size(4)))  # torch.Size([8, 7, 24, 24])
        cha_features = self.cha_block(cha)  # torch.Size([8, 7, 144, 144])

        weights_sst = self.sst_projection(sst_features.view(sst_features.size(0), sst_features.size(1), -1))  # torch.Size([8, 7, 1])
        weights_sss = self.sss_projection(sss_features.view(sss_features.size(0), sss_features.size(1), -1))  # torch.Size([8, 7, 1])
        weights_ssh = self.ssh_projection(ssh_features.view(ssh_features.size(0), ssh_features.size(1), -1))  # torch.Size([8, 7, 1])
        weights_cha = self.cha_projection(cha_features.view(cha_features.size(0), cha_features.size(1), -1))  # torch.Size([8, 7, 1])
        weights_cur = self.cur_projection(curr_features.view(curr_features.size(0), curr_features.size(1), -1))  # torch.Size([8, 7, 1])

        weights = torch.cat([weights_sst, weights_sss, weights_ssh, weights_cha, weights_cur], dim=-1)  # torch.Size([8, 7, 5])
        weights = torch.tanh(weights)  # torch.Size([8, 7, 5])
        weights = F.softmax(weights, dim=-1)  # torch.Size([8, 7, 5])
        sst_features = sst_features * weights[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # torch.Size([8, 7, 24, 24])
        sss_features = sss_features * weights[:, :, 1].unsqueeze(-1).unsqueeze(-1)  # torch.Size([8, 7, 24, 24])
        ssh_features = ssh_features * weights[:, :, 2].unsqueeze(-1).unsqueeze(-1)  # torch.Size([8, 7, 24, 24])
        curr_features = curr_features * weights[:, :, 3].unsqueeze(-1).unsqueeze(-1)  # torch.Size([8, 7, 24, 24])
        cha_features = cha_features * weights[:, :, 4].unsqueeze(-1).unsqueeze(-1)  # torch.Size([8, 7, 144, 144])

        earlybird_features = self.earlybird(earlybird_heatmap)  # torch.Size([8, 7, 48, 48])
        heatmap_features = self.heatmap(heatmap)  # torch.Size([8, 7, 48, 48])

        sst_features = self.sea_surface_split(sst_features)  # torch.Size([8, 7, 36, 16])
        sss_features = self.sea_surface_split(sss_features)  # torch.Size([8, 7, 36, 16])
        ssh_features = self.sea_surface_split(ssh_features)  # torch.Size([8, 7, 36, 16])
        curr_features = self.sea_surface_split(curr_features)  # torch.Size([8, 7, 36, 16])
        earlybird_features = self.heat_split(earlybird_features)  # torch.Size([8, 7, 36, 64])
        heatmap_features = self.heat_split(heatmap_features)  # torch.Size([8, 7, 36, 64])
        cha_features = self.cha_split(cha_features)  # torch.Size([8, 7, 36, 576])

        return sst_features, sss_features, ssh_features, curr_features, cha_features, earlybird_features, heatmap_features


class Encoder_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class Encoder_FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Encoder_Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim_head = int(dim / heads)
        self.heads = heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)


        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_projection = nn.Linear(patch_sea_surface_width * patch_sea_surface_width * 4 + patch_cha_width * patch_cha_height + patch_heat_width * patch_heat_height * 2, d_model)
        self.purple_pos_emb = nn.Parameter(torch.randn(1, patch_count, d_model)).to(device)
        self.yellow_pos_emb = nn.Parameter(torch.randn(1, output_days, d_model * 2)).to(device)

        self.yellow_layers = nn.ModuleList([])
        for _ in range(blocks):
            self.yellow_layers.append(nn.ModuleList([
                Encoder_PreNorm(d_model, Encoder_Attention(d_model, heads=n_heads)),
                Encoder_PreNorm(d_model, Encoder_FeedForward(d_model, d_model * 2))
            ]))

        self.from_yellow_to_gray_features = nn.Linear(d_model * patch_count, d_model * 2)

        self.gray_layers = nn.ModuleList([])
        for _ in range(blocks):
            self.gray_layers.append(nn.ModuleList([
                Encoder_PreNorm(d_model * 2, Encoder_Attention(d_model * 2, heads=8)),
                Encoder_PreNorm(d_model * 2, Encoder_FeedForward(d_model * 2, d_model * 4))
            ]))

    def forward(self, enc_in_features):
        # torch.Size([8, 7, 36, 768])
        # sst sss ssh cha curr early all

        embed_features = self.linear_projection(enc_in_features)  # torch.Size([8, 7, 36, d_model])
        yellow_feature = torch.zeros_like(embed_features).to(device)  # torch.Size([8, 7, 36, d_model])
        for i in range(embed_features.size(1)):
            pink_feature = embed_features[:, i] + self.purple_pos_emb  # [8, 1, 36, d_model] + [1, 36, d_model] = [8, 36, d_model]
            ## yellow_Encoder
            for attn, ff in self.yellow_layers:
                pink_feature = attn(pink_feature) + pink_feature  # torch.Size([8, 36, d_model])
                pink_feature = ff(pink_feature) + pink_feature  # torch.Size([8, 36, d_model])
            yellow_feature[:, i] = pink_feature  # torch.Size([8, 1, 36, d_model])

        # [8, 7, 36, d_model] ===> [8, 7, 36*d_model] ===> ([8, 7, d_model*2])
        yellow_feature = self.from_yellow_to_gray_features(yellow_feature.view(yellow_feature.size(0), yellow_feature.size(1), -1))

        enc_out_feature = yellow_feature + self.yellow_pos_emb  # [8, 7, d_model*2]
        ## gray_encoder
        for attn, ff in self.gray_layers:
            enc_out_feature = attn(enc_out_feature) + enc_out_feature  # torch.Size([8, 7, d_model*2])
            enc_out_feature = ff(enc_out_feature) + enc_out_feature  # torch.Size([8, 7, d_model*2])
        return enc_out_feature  # torch.Size([8, 7, d_model*2])


class Decoder_Heat_FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channel * 2, kernel_size=(3, 3), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(conv_channel * 2)
        self.ResBlock_1 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
            ResBlock(in_channels=conv_channel * 2, out_channels=conv_channel * 2, kernel_size=3),
        )

        self.conv2 = nn.Conv2d(in_channels=conv_channel * 2, out_channels=conv_channel * 4, kernel_size=(3, 3), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(conv_channel * 4)

        self.ResBlock_2 = nn.Sequential(
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
            ResBlock(in_channels=conv_channel * 4, out_channels=conv_channel * 4, kernel_size=3),
        )

        self.conv3 = nn.Conv2d(in_channels=conv_channel * 4, out_channels=1, kernel_size=(3, 3), padding='same', bias=False)

    def forward(self, inputs):
        features = F.relu(self.bn1(self.conv1(inputs)))
        features = self.ResBlock_1(features)
        features = F.relu(self.bn2(self.conv2(features)))
        features = self.ResBlock_2(features)
        features = self.conv3(features)
        return features


class Decoder_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, xq, xk, xv, mask):
        return self.fn(self.norm(xq), self.norm(xk), self.norm(xv), mask)


class Decoder_FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, xq, xk, xv, mask):
        return self.net(xq)


class Decoder_Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim_head = int(dim / heads)
        self.heads = heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, in_q, in_k, in_v, mask):
        q = rearrange(self.to_q(in_q), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(in_k), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(in_v), 'b n (h d) -> b h n d', h=self.heads)
        mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots.masked_fill_(mask.bool(), -1e9)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.heatmap = Decoder_Heat_FeatureExtraction()
        self.heat_split = Rearrange('b o (w p1) (h p2) -> b o (w h) (p1 p2)',
                                    p1=patch_heat_width,
                                    p2=patch_heat_height).to(device)

        self.linear_projection = nn.Linear(patch_heat_width * patch_heat_height, d_model)
        self.purple_pos_emb = nn.Parameter(torch.randn(1, patch_count, d_model)).to(device)
        self.green_pos_emb = nn.Parameter(torch.randn(1, output_days, d_model * 2)).to(device)

        self.green_layers = nn.ModuleList([])
        for _ in range(blocks):
            self.green_layers.append(nn.ModuleList([
                Encoder_PreNorm(d_model, Encoder_Attention(d_model, heads=8)),
                Encoder_PreNorm(d_model, Encoder_FeedForward(d_model, d_model * 2))
            ]))

        self.from_green_to_yellow_features = nn.Linear(d_model * patch_count, d_model * 2)

        self.decoder_self_attn = nn.ModuleList([])
        for _ in range(blocks):
            self.decoder_self_attn.append(nn.ModuleList([
                Decoder_PreNorm(d_model * 2, Decoder_Attention(d_model * 2, heads=8)),  # 自注意力
                Decoder_PreNorm(d_model * 2, Decoder_Attention(d_model * 2, heads=8)),  # 交互注意力
                Encoder_PreNorm(d_model * 2, Encoder_FeedForward(d_model * 2, d_model * 4))
            ]))

        self.final_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, heat_width * heat_height),
            nn.ReLU()
        )

    def forward(self, dec_heatmap, enc_out_feature):
        # ============[8, 7, 48, 48]===[8, 7, d_model*2]======
        # =======[8, 1, 48, 48] * 7 ==> [8, 7, 48, 48] ========
        heatmap_features = torch.cat([self.heatmap(dec_heatmap[:, i].unsqueeze(1)) for i in range(dec_heatmap.size(1))], dim=1)  # torch.Size([8, 7, 48, 48])
        patch_features = self.heat_split(heatmap_features)  # torch.Size([8, 7, 36, 64])
        embed_features = self.linear_projection(patch_features)  # torch.Size([8, 7, 36, d_model])
        green_feature = torch.zeros_like(embed_features).to(device)  # torch.Size([8, 7, 36, d_model])

        for i in range(embed_features.size(1)):
            pink_feature = embed_features[:, i] + self.purple_pos_emb  # [8, 1, 36, d_model] + [1, 36, d_model] = [8, 36, d_model]
            ## orange_Encoder
            for attn, ff in self.green_layers:
                pink_feature = attn(pink_feature) + pink_feature  # [8, 36, d_model]
                pink_feature = ff(pink_feature) + pink_feature  # [8, 36, d_model]
            green_feature[:, i] = pink_feature  # [8, 1, 36, d_model]
        # [8, 7, 36, d_model] ===> [8, 7, 36*d_model] ===> ([8, 7, d_model*2])
        green_feature = self.from_green_to_yellow_features(green_feature.view(green_feature.size(0), green_feature.size(1), -1))

        dec_out_feature = green_feature + self.green_pos_emb  # ([8, 7, d_model*2])

        #                     8                     7                   d_model*2
        mask_shape = [dec_out_feature.size(0), dec_out_feature.size(1), dec_out_feature.size(1)]
        mask = np.triu(np.ones(mask_shape), k=1)  # torch.Size([16, 7, 7])  0代表可以看见，1代表看不见
        mask = torch.from_numpy(mask).byte().to(device)  # torch.Size([16, 7, 7])  0代表可以看见，1代表看不见
        mask1 = np.zeros(mask_shape)  # torch.Size([16, 7, 7])  0代表可以看见，1代表看不见
        mask1 = torch.from_numpy(mask1).byte().to(device)  # torch.Size([16, 7, 7])  0代表可以看见，1代表看不见

        for attn_self, attn_inter, ff in self.decoder_self_attn:
            dec_out_feature = attn_self(dec_out_feature, dec_out_feature, dec_out_feature, mask) + dec_out_feature  # 自注意力  ([8, 7, d_model*2])
            dec_out_feature = attn_inter(dec_out_feature, enc_out_feature, enc_out_feature, mask1) + dec_out_feature  # 交互注意力 ([8, 7, d_model*2])
            dec_out_feature = ff(dec_out_feature) + dec_out_feature  # ([8, 7, d_model*2])

        dec_out = self.final_projection(dec_out_feature)  # ([8, 7, heat_width*heat_height])
        dec_out = rearrange(dec_out, 'b d (w h) -> b d w h', h=heat_width, w=heat_height) # ([8, 7, heat_width, heat_height])
        return dec_out


class Dap(nn.Module):
    def __init__(self):
        super(Dap, self).__init__()
        self.hydro_features = Hydrological_Feature_Construction().to(device)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha):
        # Concatenate the extracted features together
        # [batch, output_day, [sst sss ssh cha curr early all], width, height]
        enc_heatmap = heatmap[:, :14]
        dec_heatmap = heatmap[:, -8:-1]
        # <=================([8, 7, 36, 16])=================>([8, 7, 36, 576])<========([8, 7, 36, 64])========>
        sst_features, sss_features, ssh_features, curr_features, cha_features, earlybird_features, heatmap_features = (
            self.hydro_features(enc_heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha))
        enc_in_features = torch.cat([sst_features, sss_features, ssh_features, curr_features, cha_features, earlybird_features, heatmap_features], dim=-1)  # torch.Size([8, 7, 36, 768])
        enc_out_features = self.encoder(enc_in_features)  # torch.Size([8, 7, d_model*2])
        dec_out = self.decoder(dec_heatmap, enc_out_features)# ([8, 7, heat_width, heat_height])
        return dec_out


if __name__ == '__main__':
    batch_size = 2
    heatmap = torch.rand(size=[batch_size, 21, heat_width, heat_height]).to(device)
    earlybird_heatmap = torch.rand(size=[batch_size, 14, heat_width, heat_height]).to(device)
    ssh = torch.rand(size=[batch_size, 14, sea_surface_width, sea_surface_height]).to(device)
    sst = torch.rand(size=[batch_size, 14, sea_surface_width, sea_surface_height]).to(device)
    sss = torch.rand(size=[batch_size, 14, sea_surface_width, sea_surface_height]).to(device)
    curr = torch.rand(size=[batch_size, 14, 3, sea_surface_width, sea_surface_height]).to(device)
    cha = torch.rand(size=[batch_size, 14, cha_width, cha_height]).to(device)

    model = Dap().to(device)
    model.hydro_features.sst_block()

    total = sum([param.nelement() for param in model.parameters()])
    print(total, total / 1e6)

    # out = model(heatmap, earlybird_heatmap, ssh, sst, sss, curr, cha)

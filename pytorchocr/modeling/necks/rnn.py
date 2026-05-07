import os, sys
import torch
import torch.nn as nn
from pytorchocr.modeling.backbones.rec_svtrnet import Block, ConvBNLayer

class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == 1
        x = x.squeeze(dim=2)
        # x = x.transpose([0, 2, 1])  # paddle (NTC)(batch, width, channels)
        x = x.permute(0,2,1)
        return x


class EncoderWithRNN_(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN_, self).__init__()
        self.out_channels = hidden_size * 2
        self.rnn1 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)
        self.rnn2 = nn.LSTM(in_channels, hidden_size, bidirectional=False, batch_first=True, num_layers=2)

    def forward(self, x):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        out1, h1 = self.rnn1(x)
        out2, h2 = self.rnn2(torch.flip(x, [1]))
        return torch.cat([out1, torch.flip(out2, [1])], 2)

# USED
class EncoderWithRNN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithRNN, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(
            in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True) # batch_first:=True

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

# USED

class EncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(EncoderWithFC, self).__init__()
        self.out_channels = hidden_size
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
            bias=True,
            )

    def forward(self, x):
        x = self.fc(x)
        return x

# USED THIS is the one used
"""

"""
class EncoderWithSVTR(nn.Module):
    # Lightweight transformer neck used inside MultiHead for PP-OCRv5.
    # Takes the 2D feature map from the backbone, applies global self-attention
    # over all spatial positions, then merges the attended features back with
    # the original map via a guide shortcut before flattening to a sequence.
    #
    # For PP-OCRv5_server_rec the config is:
    #   in_channels=2048, dims=120, depth=2, hidden_dims=120,
    #   kernel_size=[1,3], use_guide=True
    #
    # Shape trace (inference):
    #   in:      [B, 2048, 1, 40]
    #   conv1:   [B,  256, 1, 40]   ConvBN(2048→256, k=[1,3], swish)
    #   conv2:   [B,  120, 1, 40]   ConvBN(256→120,  k=1,     swish)
    #   flatten: [B,   40, 120]
    #   blocks:  [B,   40, 120]     2× SVTR Global Block (see rec_svtrnet.py::Block)
    #   norm:    [B,   40, 120]     LayerNorm
    #   conv3:   [B, 2048, 1, 40]   ConvBN(120→2048, k=1, swish)  – restore channels
    #   cat:     [B, 4096, 1, 40]   cat(guide h, conv3)  – guide shortcut
    #   conv4:   [B,  256, 1, 40]   ConvBN(4096→256, k=3)
    #   conv1x1: [B,  120, 1, 40]   ConvBN(256→dims=120, k=1)
    #   Im2Seq:  [B,   40, 120]     squeeze H=1, permute → sequence for CTC
    #
    # LoRA targets inside the SVTR blocks (indexed as svtr_block[i]):
    #   .mixer.qkv   Linear(120→360)   key/query/value projection
    #   .mixer.proj  Linear(120→120)   output projection
    #   .mlp.fc1     Linear(120→240)   MLP expand
    #   .mlp.fc2     Linear(240→120)   MLP contract
    def __init__(
            self,
            in_channels,
            dims=64,  # XS
            depth=2,
            hidden_dims=120,
            use_guide=False,
            num_heads=8,
            qkv_bias=True,
            mlp_ratio=2.0,
            drop_rate=0.1,
            kernel_size=[3,3],
            attn_drop_rate=0.1,
            drop_path=0.,
            qk_scale=None):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth # depth is 2
        self.use_guide = use_guide # set to True
        self.conv1 = ConvBNLayer(
            in_channels,#2048
            in_channels // 8,
            kernel_size=kernel_size,# [1,3]
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act='swish')
        self.conv2 = ConvBNLayer(
            in_channels // 8,
            hidden_dims,
            kernel_size=1,
            act='swish')
            # hidden dims 120
            # kernel size 1
        self.svtr_block = nn.ModuleList([
            Block(
                dim=hidden_dims,#120
                num_heads=num_heads,#8
                mixer='Global',
                HW=None,
                mlp_ratio=mlp_ratio,#2.0
                qkv_bias=qkv_bias,#True
                qk_scale=qk_scale,# None
                drop=drop_rate,#0.1
                act_layer='swish',
                attn_drop=attn_drop_rate,# 0.1
                drop_path=drop_path,#0.0
                norm_layer='nn.LayerNorm',
                epsilon=1e-05,
                prenorm=False) for i in range(depth)# depth is set to 2
        ])
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)#120
        self.conv3 = ConvBNLayer(
            hidden_dims, in_channels, kernel_size=1, act='swish')
        # in channels 2048
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(
            2 * in_channels, in_channels // 8, padding=1, act='swish')
        # go up from 2048*2 then go to 256

        self.conv1x1 = ConvBNLayer(
            in_channels // 8, dims, kernel_size=1, act='swish')#120
        self.out_channels = dims#120
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Interesting information about init weights
        """
        # weight initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # use_guide=True: keep a detached copy of the backbone output as the
        # shortcut (guide) so gradients don't flow back through it a second time.
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x
        h = z  # shortcut for the residual merge at the end
        # Reduce channels so the transformer runs on a compact dim (hidden_dims=120)
        z = self.conv1(z)
        z = self.conv2(z)
        # Flatten spatial dims and run global self-attention
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        for blk in self.svtr_block:
            z = blk(z)

        z = self.norm(z)
        # Restore spatial layout, project back to in_channels, merge with guide
        z = z.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)   # [B, 2*in_channels, H, W]
        z = self.conv1x1(self.conv4(z))  # → [B, dims, H, W]

        return z

# we are here
class SequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == 'reshape':
            self.only_reshape = True
        else:
            # we have the svtr encoder
            # we are using these 3 classes
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN,
                'svtr': EncoderWithSVTR,
            }
            # double check that svtr is inside the dictionary
            assert encoder_type in support_encoder_dict, '{} must in {}'.format(
                encoder_type, support_encoder_dict.keys())
            # we use svtr in here
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, **kwargs)
            else:
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size)
            # repeated ?
            self.out_channels = self.encoder.out_channels
            # set to false
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != 'svtr':
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            # we are here this is hte one we are using
            x = self.encoder(x)
            # reshaping
            x = self.encoder_reshape(x)
            return x
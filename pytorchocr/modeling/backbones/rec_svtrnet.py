import torch
import torch.nn as nn
from pytorchocr.modeling.common import Activation
import numpy as np
"""
This function randomly drops entire samples during training as a regularization technique.
Intuition
Standard DropoutDrop PathWhat's droppedIndividual neuronsEntire residual branchesGranularityPer elementPer sampleUsed inDense layersResidual blocks / transformers
It's essentially saying: 
"for this training step, pretend this residual branch didn't exist for some samples in the batch." 
This forces the network to not over-rely on any single layer.
"""
def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    # If no dropping is needed (prob = 0 or we're in inference mode), return input unchanged.
    if drop_prob == 0. or not training:
        return x
    # If drop_prob = 0.2, then keep_prob = 0.8 — meaning 80% of samples survive.
    keep_prob = torch.as_tensor(1 - drop_prob)
    # Creates a shape like (batch_size, 1, 1, 1).
    # The 1s are intentional — the mask applies per sample, not per element,
    # so it broadcasts across all other dimensions.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    # torch.rand gives values in [0, 1)
    # Adding keep_prob shifts them to [keep_prob, 1 + keep_prob)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    # floor() binarizes: values ≥ 1 → 1 (keep), values < 1 → 0 (drop)
    # The probability of a value being ≥ 1 is exactly keep_prob ✓
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    # Multiplying by random_tensor zeros out dropped samples
    # Dividing by keep_prob rescales surviving samples to maintain the expected value — same idea as standard dropout
    return output

# USED
class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act='gelu'):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = Activation(act_type=act, inplace=True)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

# USED
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# USED
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

# USED
class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='gelu',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = Activation(act_type=act_layer, inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# USED
"""
This is a convolution-based alternative to local attention.
 It does the same job as mixer='Local' in the previous Attention class, but uses a Conv2d instead of the expensive Q@K attention math.

  }
The whole trick of this class is a shape shuffle: 
it temporarily pretends the token sequence is a 2D image so it can run a cheap Conv2d, 
then shuffles back. Here's the full flow in plain English:
The whole trick of this class is a shape shuffle: it temporarily pretends the token sequence is a 2D image so it can run a cheap Conv2d, then shuffles back. Here's the full flow in plain English:
__init__ builds one Conv2d with groups=num_heads. That groups parameter is the key — it splits channels into num_heads independent groups, each with its own kernel, mimicking how attention heads work independently.
forward does three things:

(B, N, C) → reshape → (B, C, H, W) — turn the flat token list into a 2D grid
Run the Conv2d — each token gathers info from its local_k × local_k neighborhood
(B, C, H, W) → flatten + permute → (B, N, C) — back to token sequence

The core insight from the comparison tab: 
ConvMixer and 
Local Attention do the same conceptual job (restrict each token to its neighbors) but via completely different mechanics.
Conv uses a learned fixed kernel sliding across the grid. Local Attention computes Q@K scores then masks out distant ones.
Conv is faster but blind to content — the kernel weights don't change based on what the tokens contain. 
Attention is slower but adaptive — which neighbors matter depends on the actual input.
ConvMixer
Local Attention
Complexity
    O(N × k²)
    O(N²)
Learned weights
    small fixed kernel (e.g. 3×3×C)
    Q, K, V projections
How scope is limited
    kernel size hard-limits window
    -inf mask blocks far tokens
Attention weights?
    No — fixed kernel weights
    Yes — content-dependent
Speed
    Fast (GPU-optimised conv)
    Slower (matmul + softmax)
Best for
    early/shallow layers
    deeper layers needing context
"""
"""
How does grouping work?
The groups parameter controls how many independent lanes the convolution is split into. Click through the three tabs to see the spectrum.

Tab 1 — Normal conv (groups=1): every output channel's kernel sees every input channel. If you have 8 input and 8 output channels, each kernel has depth 8. They all share the full input — very expressive, but expensive.

Tab 2 — Grouped conv (groups=4): channels are divided into 4 color-coded groups. Blue input channels only connect to the blue kernel, which only produces blue output channels. Red talks to red. They never cross. Each kernel is now depth 2 instead of 8 — 4× fewer parameters, 4× faster.

Tab 3 — Depthwise (groups=C): the extreme end — every single channel gets its own private kernel of depth 1. Zero channel mixing at all. This is what MobileNet uses to be ultra-lightweight, but it needs a follow-up 1×1 conv afterwards to let channels communicate again.

In ConvMixer, groups=num_heads maps directly to the attention head concept — each head gets its own slice of channels and learns its own spatial pattern independently. Head 1 might learn to detect horizontal edges, head 2 vertical strokes, head 3 curves — all without interfering with each other, exactly like attention heads specialising on different aspects of the input.


"""
class ConvMixer(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            HW=[8, 25],
            local_k=[3, 3], ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,
            1, [local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
            )
        # Feature map (B, C, H, W) — each cell = one token position
        # Conv2d parameters explained
        # in_channels = dim
            # input feature depth
        # out_channels = dim
            # same depth out
        # kernel_size = local_k
            # [3,3] window size
        # stride = 1
            # slide one step at a time
        # padding = [k//2, k//2]
            # keep H,W unchanged
        # groups = num_heads
        # ← key param
            # each head gets dim/num_heads channels
            # with its own independent kernel
            # = depthwise-style, one kernel per head

    def forward(self, x):
        # input is (B, N, C)
        h = self.HW[0]
        w = self.HW[1]
        # you transpose it to  (B, C, N) → reshape → (B, C, H, W)
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x

# USED
"""
This is a Vision Transformer attention module with two modes: global attention (every token sees every other token) and 
local attention (each token only sees its neighbors). Let me break it down piece by piece.
what is difference between gllobal and local ?
Global vs Local is the key design choice. 
Global lets every token see every other token — full expressiveness but O(N²) cost. 
Local restricts each token to a spatial window (controlled by local_k)
, which is much cheaper and makes sense for text/OCR images where a patch mostly cares about its neighbors,
 not patches on the other side of the image.
"""
class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=[8, 25],
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        #QKV PROJECTION GIVE
        # Q what i am looking for ?
        # K what do i contain?
        # V what do i send if matched
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            """
            The local mask is built once in __init__ (not every forward pass).
             It marks positions outside the local window with -inf.
              When added to the raw attention scores before softmax, those positions become exactly zero — invisible to the attention. 
              This is critical for OCR/text image tasks where nearby patches are more relevant than distant ones.
            """
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(H * W, H + hk - 1, W + wk - 1, dtype=torch.float32)
            """
            This exact tensor structure is almost always used in
            local self-attention, 
            sliding-window attention, or
            neighborhood attention mechanisms (like those found in Swin Transformers or NAT).
            """
            for h in range(0, H):
                for w in range(0, W):
                    #  It loops through every single pixel coordinate $(h, w)$ in your $H \times W$ image grid.
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
                    # For that specific pixel,
                    # it carves out a 2D local window of height hk and width wk
                    # in the padded key/value space and sets those values to 0..
                    # Every pixel's row has a small block of 0.
                    # values (its allowed neighbors) surrounded by 1. values (the forbidden distant pixels).
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                               2].flatten(1)
            # Because the original tensor was padded to prevent edge errors,
            # the local windows are offset. This slicing removes the extra padding borders and
            # shifts the grid so that each pixel sits exactly at the center of its own local window.
            mask_inf = torch.full([H * W, H * W], fill_value=float("-Inf"), dtype=torch.float32)
            # # Adding $-\infty$ to a score drives it to zero after softmax ($\exp(-\infty) = 0$), completely blocking communication.
            # # torch.where performs this swap:
            # # Wherever mask_paddle < 1 (which are the 0. values), it keeps the 0..
            # # Wherever it is 1. (the distant pixels), it replaces it with -Inf
            # # Wherever mask_paddle is 0 (less than 1), keep 0. Otherwise, inject -inf.
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze(0).unsqueeze(1)
            # go up to 1,1,N,N
            # self.mask = mask[None, None, :]
        self.mixer = mixer

    def forward(self, x):
        # the input is (B, N, C)
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape
        qkv = self.qkv(x) # give me an output of (B, N, C×3)
        qkv = qkv.reshape((-1, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        # reshape to (B,N,3,heads,head_dim) then (3,B,heads,N,d)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        # right now q is (B, heads, N, d)
        # k is  (B, heads, N, d)
        # v is (B, heads, N, d)
        attn = (q.matmul(k.permute(0, 1, 3, 2)))
        # raw scores
        # (B, h, N, N)
        # The N×N score matrix is the cost — every token attends to every other token.
        # For a 200-token sequence that's 40,000 values per head.
        # This is why transformers are expensive.
        # The scale factor (head_dim⁻⁰·⁵) prevents dot products from getting so large that softmax saturates.
        if self.mixer == 'Local':
            # i need to understand what does local mean ???
            attn += self.mask
        # apply soft max
        attn = nn.functional.softmax(attn, dim=-1)
        # apply the attention drop
        attn = self.attn_drop(attn)
        # in here we do the following
        # (B,h,N,d) is output after the multplica then then we permute (B,N,h,d) then reshape to (B,N,C)
        x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape((-1, N, C))
        # After multiplying attention weights by V, each head has its own result. permute + reshape concatenates all heads back into one tensor.
        x = self.proj(x)
        # Then self.proj (another Linear) lets heads communicate and mix — without it, heads are completely independent.
        x = self.proj_drop(x)
        # Output shape is identical to input: (B, N, C).
        return x

# USED
# This is the transformer block that we are working with

class Block(nn.Module):
    def __init__(self,
                 dim,# 120
                 num_heads,#8
                 mixer='Global',#global
                 local_mixer=[7, 11], #[7,11]
                 HW=None,# None
                 mlp_ratio=4.,# mlp ratio = 2
                 qkv_bias=False,# qkv sset to true
                 qk_scale=None,#None
                 drop=0.,#0.1
                 attn_drop=0.,#0.1
                 drop_path=0.,#0
                 act_layer='gelu',# swich
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            # Sub-block 1 — Mixer (attention): in a classic transformer this is always Multi-Head Self-Attention.
            # Here it's made flexible — it can be Global attention,
            # Local attention, or Conv. But the job is the same: let tokens communicate with each other.
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            # LP (feed-forward network): this is identical to every transformer.
            # It expands the features to dim × mlp_ratio (so 120 × 2 = 240 here),
            # applies an activation (gelu or swish), then shrinks back.
            # It runs independently on each token with no cross-token mixing.
            x = self.norm2(x + self.drop_path(self.mlp(x)))

        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        # Both sub-blocks are wrapped in the same residual pattern: x = x + drop_path(sub_block(x)).
        # That + x is the skip connection from our earlier residual network discussion — and drop_path
        # is exactly the function you saw at the very start, zeroing entire samples to regularize.
        return x

# Used
"""
This is the entry point of the Vision Transformer — it converts a raw image into a sequence of tokens that the transformer blocks can process.
This class solves a fundamental mismatch: a raw image is a 3D cube (channels, height, width) but a transformer needs a 1D sequence of vectors. PatchEmbed bridges that gap.
Two modes, same goal:
POPE mode (the main one) uses 2 or 3 stacked ConvBNLayer blocks, each with stride=2. Every stride-2 conv halves the spatial size while expanding channels. With sub_num=2, two halvings means dividing H and W by 4 — so a 32×100 image becomes an 8×25 feature map, giving 200 patches. The key insight is gradual downsampling: going 3→embed in two steps preserves more texture and edge detail than one brutal jump, which matters a lot for text recognition where characters are tiny.
Linear mode is the original ViT approach — one single Conv2d where kernel_size = stride = patch_size. This tiles the image into non-overlapping blocks and projects each block directly. Simpler but loses fine-grained detail.
The forward is just three lines (click the last tab to see each shape): run proj, then flatten(2) to collapse H'×W' into one N dimension, then permute(0,2,1) to swap channels and tokens into the (B, N, embed_dim) shape the transformer expects.
After this class, the image is gone — there's just a sequence of 200 vectors, and the transformer blocks have no idea they came from pixels. That's the whole point.
"""
"""
Here's what each mode is really doing and why they differ:
pope sub_num=2 — two strided convolutions (each halves H and W), so you go from 3×32×100 to embed×8×25. That gives 8×25 = 200 tokens. Each convolution applies BatchNorm + GELU, so the network learns local spatial features progressively before handing anything to the transformer.
pope sub_num=3 — same idea but three strides, so the spatial dimensions get halved three times (÷8 total). From 3×32×100 you end up at embed×4×12, giving only 4×12 = 48 tokens. This is a much shorter sequence — useful if you want a faster/lighter transformer at the cost of losing fine-grained spatial detail.
linear — one single Conv2d with kernel_size=patch_size and stride=patch_size, which is mathematically equivalent to cutting the image into non-overlapping patches and linearly projecting each one. No BN, no activation, no feature hierarchy — just a flat projection. It's the classic ViT-style patchification.
The key tradeoff in a nutshell:
Sequence lengthFeature richnessSpeedpope sub_num=2200High (hierarchical)Mediumpope sub_num=348Very compressedFastestlinear200Low (flat projection)Simple
The forward is identical across all three — .proj(x).flatten(2).permute(0,2,1) — because they all produce a (B, embed, H', W') feature map, and those last two ops collapse H'×W' into tokens and reorder into (B, N, embed) as the transformer expects.

"""
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=[32, 100],
                 in_channels=3,
                 embed_dim=768,
                 sub_num=2,
                 patch_size=[4, 4],
                 mode='pope',
                 ):
        """
        This class solves a fundamental mismatch: a raw image is a 3D cube (channels, height, width)
        but a transformer needs a 1D sequence of vectors. PatchEmbed bridges that gap.

        """
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                # The forward is just three lines (click the last tab to see each shape): run proj,
                # then flatten(2) to collapse H'×W' into one N dimension, then permute(0,2,1) to swap channels and tokens into the (B, N, embed_dim) shape the transformer expects.
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act='gelu',
                        bias_attr=True),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act='gelu',
                        bias_attr=True))
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act='gelu',
                        bias_attr=True),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act='gelu',
                        bias_attr=True),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act='gelu',
                        bias_attr=True))
        elif mode == 'linear':
            self.proj = nn.Conv2d(
                1, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({}*{}) doesn't match model ({}*{}).".format(
                H,W,self.img_size[0],self.img_size[1]
            )
        # The forward is just three lines (click the last tab to see each shape):
        # run proj,
        # then flatten(2) to collapse H'×W' into one N dimension,
        # then permute(0,2,1) to swap channels and tokens into the (B, N, embed_dim) shape the transformer expects.
        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x

# USED
class SubSample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 types='Pool',
                 stride=[2, 1],
                 sub_norm='nn.LayerNorm',
                 act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                )
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):

        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))
        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


class SVTRNet(nn.Module):
    def __init__(
            self,
            img_size=[32, 100],
            in_channels=3,
            embed_dim=[64, 128, 256],
            depth=[3, 6, 3],
            num_heads=[2, 4, 8],
            mixer=['Local'] * 6 + ['Global'] *
            6,  # Local atten, Global atten, Conv
            local_mixer=[[7, 11], [7, 11], [7, 11]],
            patch_merging='Conv',  # Conv, Pool, None
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            last_drop=0.0,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer='nn.LayerNorm',
            sub_norm='nn.LayerNorm',
            epsilon=1e-6,
            out_channels=192,
            out_char_num=25,
            block_unit='Block',
            act='gelu',
            last_stage=True,
            sub_num=2,
            prenorm=True,
            use_lenhead=False,
            **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)

        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=act,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0]:depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=act,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[1])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mixer=mixer[depth[0] + depth[1]:][i],
                HW=HW,
                local_mixer=local_mixer[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=act,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[2])
        ])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False)
            self.hardswish = Activation('hard_swish', inplace=True) #nn.Hardswish()
            # self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")
            self.dropout = nn.Dropout(p=last_drop)
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], eps=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = Activation('hard_swish', inplace=True)# nn.Hardswish()
            self.dropout_len = nn.Dropout(
                p=last_drop)

        torch.nn.init.xavier_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[2], h, self.HW[1]]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        if self.use_lenhead:
            return x, len_x
        return x
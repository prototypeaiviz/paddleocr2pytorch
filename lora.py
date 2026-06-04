"""
LoRA (Low-Rank Adaptation) implementation for fine-tuning pre-trained models.

LoRA decomposes weight updates as: ΔW = A @ B.T where A ∈ ℝ^(d_out × r) and B ∈ ℝ^(d_in × r).
Only A and B are trainable; the original weights remain frozen.

For a Linear layer with weight W ∈ ℝ^(d_out × d_in):
  output = (W + α/r * A @ B.T) @ input

where α is a scaling factor and r is the rank.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import re

import math

class LoRAConv2d(nn.Module):
    """
    Conv2d + LoRA. Wraps a frozen Conv2d and adds a low-rank conv branch:
        out = base(x) + scale * up(down(x))
    - down: in_ch -> r, replicates base kernel/stride/padding/dilation so spatial dims match
    - up:   r -> out_ch, 1x1, zero-init (so the adapter is a no-op at initialization)
    """
    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.scale = alpha / r
        self.dropout_fn = nn.Dropout(dropout)

        self.base = base                      # kept as a frozen child
        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_down = nn.Conv2d(
            base.in_channels, r,
            kernel_size=base.kernel_size, stride=base.stride,
            padding=base.padding, dilation=base.dilation,
            groups=1, bias=False,             # full low-rank mixing
        )
        self.lora_up = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)   # ΔW = 0 initially

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_up(self.lora_down(self.dropout_fn(x)))
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Wraps a frozen pre-trained Linear layer and adds trainable low-rank matrices.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge: bool = False,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            r: LoRA rank (default 8)
            alpha: Scaling factor for LoRA updates (default 16)
            dropout: Dropout rate applied to LoRA-A (default 0)
            merge: If True, merge LoRA into weight matrix (inference optimization)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.dropout_fn = nn.Dropout(dropout)
        self.merge = merge

        # LoRA weight matrices
        # A: (out_features, rank) - initialized from Gaussian
        # B: (in_features, rank) - initialized to zero (so initial ΔW = 0)
        self.lora_a = nn.Parameter(torch.randn(out_features, r))
        self.lora_b = nn.Parameter(torch.zeros(in_features, r))

        # Scaling factor: α / r (matches HuggingFace PEFT implementation)
        self.scale = alpha / r

        # Original weight and bias as nn.Parameter (non-trainable)
        self.register_buffer("weight", torch.zeros(out_features, in_features))
        self.register_buffer("bias", torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = x @ (W.T + scale * B @ A.T) + bias
                             = x @ W.T + scale * x @ B @ A.T + bias
        """
        # Ensure all tensors are on the same device as x

        # Original linear transformation
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        # LoRA adaptation: scale * (x @ B @ A.T)
        # x: (batch, ..., in_features)
        # B: (in_features, rank)
        # x @ B: (batch, ..., rank)
        # A: (out_features, rank) -> A.T: (rank, out_features)
        # (x @ B) @ A.T: (batch, ..., out_features)
        x_dropped = self.dropout_fn(x)
        lora_update = (x_dropped @ self.lora_b) @ self.lora_a.T
        lora_update = lora_update * self.scale
        out = out + lora_update

        return out


class LoRAInjector:
    """
    This function is a surgeon — it walks into the model, finds specific Linear layers by their path name,
    swaps them out for LoRA-wrapped versions, then freezes everything else.

    Utility class to inject LoRA layers into a pre-trained model.
    Replaces target Linear layers with LoRA-wrapped versions while freezing the backbone.
    """

    @staticmethod
    def inject_lora(
        model: nn.Module,
        target_modules: List[str],
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_backbone: bool = True,
        unfreeze_norms: bool = True,

    ) -> nn.Module:
        """
        Inject LoRA layers into specified modules.

        Args:
            model: The model to adapt
            target_modules: List of module names to apply LoRA to (dot-separated paths)
            r: LoRA rank
            alpha: Scaling factor
            dropout: Dropout rate
            freeze_backbone: If True, freeze all non-LoRA parameters
            unfreeze_norms: if True, unfreeze norms
        Returns:
            Model with LoRA injected
        """
        # Track which modules were updated for verification
        updated_modules = []

        for target_name in target_modules:
            parts = target_name.split('.')# split based on the dot
            module = model

            # Navigate to parent module
            #give me everything except hte last part
            found = True
            for part in parts[:-1]:
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    print(f"Warning: Could not find path {target_name}, skipping")
                    found = False
                    break
            if not found:
                continue

            # Get the target linear layer
            final_part = parts[-1]
            if hasattr(module, final_part):
                # this is the part that i am most interested in implementing lora ON
                linear_layer = getattr(module, final_part)

                if isinstance(linear_layer, nn.Linear):
                    # Create LoRA-wrapped version
                    # cchecking that this is a linar layer
                    lora_linear = LoRALinear(
                        in_features=linear_layer.in_features,
                        out_features=linear_layer.out_features,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                    )

                    # Transfer original weight and bias to LoRA layer
                    # Copy data into the registered buffers
                    lora_linear.weight.data.copy_(linear_layer.weight.data)
                    if linear_layer.bias is not None:
                        lora_linear.bias.data.copy_(linear_layer.bias.data)
                    else:
                        lora_linear.bias.data.zero_()

                    # Replace the original layer
                    setattr(module, final_part, lora_linear)
                    # setattr(obj, name, value) is identical to obj.name = value — it writes one.
                    updated_modules.append(target_name)

                elif isinstance(linear_layer, nn.Conv2d):  # ← NEW
                    setattr(module, final_part,
                            LoRAConv2d(linear_layer, r=r, alpha=alpha, dropout=dropout))
                    updated_modules.append(target_name)

                else:
                    print(f"Skip {target_name}: {type(linear_layer).__name__} not supported")

        print(f"✓ Injected LoRA into {len(updated_modules)} modules")
        for name in updated_modules:
            print(f"  - {name}")

        # Freeze backbone (all non-LoRA parameters)
        if freeze_backbone:
            LoRAInjector.freeze_non_lora_parameters(model, unfreeze_norms=unfreeze_norms)

        return model

    @staticmethod
    def freeze_non_lora_parameters(model: nn.Module, unfreeze_norms: bool = True) -> None:
        """
        Freeze all parameters except LoRA parameters and LayerNorm layers.
        """
        frozen_count = 0
        lora_count = 0
        norm_count = 0

        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                lora_count += 1

            # Simplified and robust check for any normalization layer within the processing heads or encoders
            elif (unfreeze_norms
                  and ('norm' in name.lower() or 'ln' in name.lower())
                  and ('head' in name.lower() or 'encoder' in name.lower() or 'svtr_block' in name.lower())
                  and ('weight' in name or 'bias' in name)):
                param.requires_grad = True
                norm_count += 1

            else:
                param.requires_grad = False
                frozen_count += 1

        print(f"✓ Froze          {frozen_count} parameters")
        print(f"✓ LoRA           {lora_count} parameters are trainable")
        print(f"✓ LayerNorm      {norm_count} parameters are trainable")
        print(f"✓ Total trainable: {lora_count + norm_count}")

    @staticmethod
    def get_lora_stats(model: nn.Module) -> dict:
        """
        Get statistics about LoRA parameters in the model.
        """
        lora_params = 0
        total_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lora_parameters': lora_params,
            'lora_percentage': (lora_params / trainable_params * 100) if trainable_params > 0 else 0,
        }

def backbone_spatial_targets(model, stages=(2,)):
    """Depthwise/spatial convs (the shape filters) in the given stages.
       Skips 1x1 channel-mixers and the downsample convs."""
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        mt = re.search(r'stages\.(\d+)\.', name)
        if not mt or int(mt.group(1)) not in stages:
            continue
        if m.kernel_size != (1, 1) and 'downsample' not in name:
            out.append(name)              # stage2 -> the .conv2.conv (5x5 depthwise)
    return out
def inject_lora_to_ppocr_v5_with_backbone(
        model, r_head=16, alpha_head=32,
        r_backbone=8, alpha_backbone=16,        # smaller rank for the backbone
        dropout=0.0, backbone_stages=(2,), unfreeze_norms=False):

    head_targets = [
        'head.ctc_encoder.encoder.svtr_block.0.mixer.qkv',
        'head.ctc_encoder.encoder.svtr_block.0.mixer.proj',
        'head.ctc_encoder.encoder.svtr_block.0.mlp.fc1',
        'head.ctc_encoder.encoder.svtr_block.0.mlp.fc2',
        'head.ctc_encoder.encoder.svtr_block.1.mixer.qkv',
        'head.ctc_encoder.encoder.svtr_block.1.mixer.proj',
        'head.ctc_encoder.encoder.svtr_block.1.mlp.fc1',
        'head.ctc_encoder.encoder.svtr_block.1.mlp.fc2',
        'head.ctc_head.fc',
    ]
    backbone_targets = backbone_spatial_targets(model, stages=backbone_stages)

    # inject WITHOUT freezing yet (freeze once at the end)
    LoRAInjector.inject_lora(model, head_targets,     r=r_head,     alpha=alpha_head,
                             dropout=dropout, freeze_backbone=False)
    LoRAInjector.inject_lora(model, backbone_targets, r=r_backbone, alpha=alpha_backbone,
                             dropout=dropout, freeze_backbone=False)

    LoRAInjector.freeze_non_lora_parameters(model, unfreeze_norms=unfreeze_norms)
    print(LoRAInjector.get_lora_stats(model))
    return model
def inject_lora_to_ppocr_v5(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    unfreeze_norms:bool = True,
) -> nn.Module:
    """
    Convenience function to inject LoRA into PP-OCRv5_server_rec model.
    Targets the 9 key linear layers: 4 per SVTR block + 1 CTCHead.fc

    Args:
        model: BaseModel instance with MultiHead
        r: LoRA rank
        alpha: Scaling factor
        dropout: Dropout rate
        unfreeze_norms:unfreeze norms
    Returns:
        Model with LoRA injected and backbone frozen
    """
    target_modules = [
        # SVTR Block 0
        'head.ctc_encoder.encoder.svtr_block.0.mixer.qkv',
        'head.ctc_encoder.encoder.svtr_block.0.mixer.proj',
        'head.ctc_encoder.encoder.svtr_block.0.mlp.fc1',
        'head.ctc_encoder.encoder.svtr_block.0.mlp.fc2',

        # SVTR Block 1
        'head.ctc_encoder.encoder.svtr_block.1.mixer.qkv',
        'head.ctc_encoder.encoder.svtr_block.1.mixer.proj',
        'head.ctc_encoder.encoder.svtr_block.1.mlp.fc1',
        'head.ctc_encoder.encoder.svtr_block.1.mlp.fc2',

        # CTC Head
        'head.ctc_head.fc',
    ]

    print("\n" + "="*70)
    print("  Injecting LoRA into PP-OCRv5_server_rec")
    print("="*70)

    model = LoRAInjector.inject_lora(
        model=model,
        target_modules=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
        freeze_backbone=True,
        unfreeze_norms=unfreeze_norms
    )

    # Print statistics
    stats = LoRAInjector.get_lora_stats(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters:      {stats['total_parameters']:,}")
    print(f"  Trainable parameters:  {stats['trainable_parameters']:,}")
    print(f"  LoRA parameters:       {stats['lora_parameters']:,}")
    print(f"  LoRA % of trainable:   {stats['lora_percentage']:.2f}%")
    print("="*70 + "\n")

    return model


if __name__ == '__main__':
    # Quick test
    print("Testing LoRA implementation...")

    # Create a simple test
    lora_linear_model = LoRALinear(in_features=120, out_features=120, r=8)
    x = torch.randn(2, 120)
    output = lora_linear_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ LoRA layer works correctly")

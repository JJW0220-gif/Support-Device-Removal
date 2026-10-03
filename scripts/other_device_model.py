"""Extended device model with a third "Other Devices" decoder."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
CXR_MODEL_ROOT = REPO_ROOT / "cxr-lines-and-tubes"
sys.path.insert(0, str(CXR_MODEL_ROOT))

from skp.models.segmentation.native_unet import UnetDecoder  # noqa: E402
from skp.models.segmentation.smp_dual_decoder_device_unet import (  # noqa: E402
    Net as DeviceNet,
)


class DeviceNetWithOther(DeviceNet):
    """Add a third decoder for 'Other Support Devices' binary segmentation."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # Create Other decoder: same architecture as agnostic decoder but 1-channel output
        initialized_cfg = copy.deepcopy(cfg)
        initialized_cfg.pretrained = False
        initialized_cfg.num_classes = 1  # binary segmentation

        # Create a NEW decoder (not deepcopy from frozen one)
        from skp.models.segmentation.native_unet import NativeUnet
        temp_model = NativeUnet(initialized_cfg)
        self.other_decoder = temp_model.decoder  # Fresh, trainable decoder

        # 1-channel output head with proper initialization
        self.other_head = nn.Sequential(
            nn.Conv2d(
                self.agnostic_head[0].in_channels,
                1,
                kernel_size=3,
                padding=1,
            )
        )
        # Initialize with small weights to prevent explosion
        nn.init.xavier_normal_(self.other_head[0].weight, gain=0.01)
        nn.init.zeros_(self.other_head[0].bias)

    def forward(self, batch, return_loss=False, return_features=False, **_):
        x = batch["x"]

        # Encoder (frozen during Other training)
        with torch.no_grad():
            features = self.encoder(x)

        # Original branches (frozen, for generating XOR labels during training)
        if self.training:
            # During training, we only need Other branch forward
            other_logits = self.other_head(self.other_decoder(features))
            out = {"other_logits": other_logits}
        else:
            # During inference, compute all branches
            agnostic_logits = self.agnostic_head(self.agnostic_decoder(features))
            family_logits = self.device_head(self.device_decoder(features))
            other_logits = self.other_head(self.other_decoder(features))

            out = {
                "agnostic_logits": agnostic_logits,
                "family_logits": family_logits,
                "other_logits": other_logits,
            }

        if return_features:
            out["features"] = features
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def freeze_original_branches(self):
        """Freeze encoder + agnostic + family decoders, keep only Other trainable."""
        for module in [self.encoder, self.agnostic_decoder, self.agnostic_head,
                       self.device_decoder, self.device_head]:
            module.requires_grad_(False)
            module.eval()
        # Explicitly ensure other_decoder and other_head are trainable
        self.other_decoder.requires_grad_(True)
        self.other_head.requires_grad_(True)
        self.other_decoder.train()
        self.other_head.train()
        self.other_decoder.requires_grad_(True)
        self.other_head.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep original branches in eval mode
        self.encoder.eval()
        self.agnostic_decoder.eval()
        self.agnostic_head.eval()
        self.device_decoder.eval()
        self.device_head.eval()
        return self

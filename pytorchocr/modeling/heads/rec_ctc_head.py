import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
This is the final layer of the entire OCR pipeline — it takes the 40 token vectors from the neck and answers "what character is at each position?"
This is actually the simplest class in the whole pipeline — almost all the heavy work was done upstream. Here's the full picture:
* What it does in one sentence: take each of the 40 token vectors (size 120) and ask "which of the 6625 characters in our vocabulary is most likely here?"
* The Linear layer does 120 → 6625 independently for every token.
    * So the output is [B, 40, 6625] — a probability grid where each row is one time-step and each column is one character.
    * The argmax of each row gives the most likely character at that position.
* The 6625 output size = 6623 real characters + 1 blank token + 1 space.
    * The blank is critical for CTC — check the second tab to see how it solves the alignment problem between 40 fixed output slots and variable-length words.
* The train vs inference split (third tab) is the most subtle part. 
    * During training, raw logits go straight to CTCLoss which internally applies log_softmax — so you must NOT softmax beforehand or the math breaks.
    * During inference there's no loss, so softmax is applied manually to turn raw scores into proper probabilities for the decoder.
* Connecting it all together — now you've seen the complete PP-OCRv5 server pipeline end to end:
    image → PatchEmbed (image to tokens) → Block layers (attention + MLP) → EncoderWithSVTR (transformer neck with guide shortcut) → CTCHead (token to character probabilities) → CTC decoder (probabilities to text).
"""
class CTCHead(nn.Module):
    # Final classification layer for CTC decoding.
    # For PP-OCRv5_server_rec: in_channels=120 (from EncoderWithSVTR dims),
    # out_channels=n_char (vocab size + 1 blank + 1 space).
    # At inference applies softmax so CTCLabelDecode gets probabilities.
    # mid_channels=None → single Linear(120→n_char).  ← LoRA target: self.fc
    def __init__(self,
                 in_channels,
                 out_channels=6625,#18385
                 fc_decay=0.0004,#1-5
                 mid_channels=None,#set to None
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,)
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats


    def forward(self, x, labels=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        return result
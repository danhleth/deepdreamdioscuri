from typing import Any, Dict

import torch
from torch import nn

class ClassificationCELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, **kwargs):
        super(ClassificationCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        pred: Dict[str, Any],
        batch: Dict[str, Any],
    ):
        pred = pred["logits"] if isinstance(pred, Dict) else pred
        target = batch["labels"] if isinstance(batch, Dict) else batch

        if pred.shape == target.shape:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target.view(-1).contiguous())

        loss_dict = {"loss": loss.item()}
        return loss, loss_dict
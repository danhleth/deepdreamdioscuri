from typing import Dict

import torch

from dioscuri.faceagegender.losses import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class BCEwithstat(torch.nn.Module):
    r"""CEwithstat is warper of binary cross-entropy loss"""

    def __init__(self, key_label='label'):
        super(BCEwithstat, self).__init__()
        self.key_label = key_label
        key_type = key_label.split('_')[-1]
        self.key_type = f"logit_{key_type}"

    def forward(self, pred, batch):
        pred = pred[self.key_type] if isinstance(pred, Dict) else pred
        pred = torch.squeeze(pred, dim=1)
        
        # in torchvision models, pred is a dict[key=out, value=Tensor]
        target = batch[self.key_label] if isinstance(batch, Dict) else batch
        # custom label is storaged in batch["mask"]
        # print("CEwithstat: pred:", pred.shape, "target:", target.shape)

        loss = torch.nn.functional.binary_cross_entropy(pred, target)
        loss_dict = {"loss": loss}
        return loss, loss_dict
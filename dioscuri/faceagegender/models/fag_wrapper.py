from torch import nn

from . import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class FAGModelWithLoss(nn.Module):
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    Example:

        from dioscuri.model.segmentation import MobileUnet
        from dioscuri.model.loss import CEwithstat
        from dioscuri.model import ModelWithLoss

        model = MobileUnet()
        loss = CEwithstat(nclasses = 2)

        modelwithloss = FAGModelWithLoss(model = model, loss = loss)
    """

    def __init__(self, model: nn.Module, criterion_age: nn.Module, criterion_gender: nn.Module):
        super().__init__()
        self.model = model
        self.criterion_age = criterion_age
        self.criterion_gender = criterion_gender

    def forward(self, batch):
        outputs = self.model(batch["input"])
        # logit_age, logit_gender = outputs['logit_age'], outputs['logit_gender']
        loss_age, _ = self.criterion_age(outputs, batch)
        loss_gender, _ = self.criterion_gender(outputs, batch)


        loss = loss_age + loss_gender
        loss_dict = {'loss':loss}

        return {
            'logit_age': outputs['logit_age'],
            'logit_gender': outputs['logit_gender'],
            'loss': loss,
            'loss_dict': loss_dict
        }

    def forward_train(self, batch):
        return self.forward(batch)

    def forward_eval(self, batch):
        return self.forward(batch)

    def state_dict(self):
        return self.model.state_dict()

    @classmethod
    def from_cfg(cls, model, criterion, getter):
        model = getter(model)
        criterion = getter(criterion)
        return cls(model, criterion)
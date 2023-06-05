from torchvision.models import resnet34

from dioscuri.base.models import MODEL_REGISTRY

from .mobilevit import mobilevit_xxs

MODEL_REGISTRY.register(resnet34)
MODEL_REGISTRY.register(mobilevit_xxs)
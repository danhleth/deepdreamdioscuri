from dioscuri.base.models import MODEL_REGISTRY

from .backbones.mobilevit import mobilevit_xs

MODEL_REGISTRY.register(mobilevit_xs)
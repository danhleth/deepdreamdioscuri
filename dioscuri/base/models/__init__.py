from dioscuri.base.registry import Registry

MODEL_REGISTRY = Registry('MODEL')

from .backbones.mobilevit import mobilevit_xxs

MODEL_REGISTRY.register(mobilevit_xxs)
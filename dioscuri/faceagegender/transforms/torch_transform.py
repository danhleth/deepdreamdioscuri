from torchvision.transforms import Normalize, Resize, ToTensor, Compose

from . import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(Normalize, prefix='')
TRANSFORM_REGISTRY.register(Resize, prefix='')
TRANSFORM_REGISTRY.register(ToTensor, prefix='')
TRANSFORM_REGISTRY.register(Compose, prefix='')
from dioscuri.base.registry import Registry

DATASET_REGISTRY = Registry('DATASET')

from .dataset import ImageDataset
from .image_dataset import IMAGEDATASET
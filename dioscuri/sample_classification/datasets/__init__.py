from dioscuri.base.datasets import DATASET_REGISTRY

from .cat_emotion_dataset import CATEMOTIONDATASET

DATASET_REGISTRY.register(CATEMOTIONDATASET)
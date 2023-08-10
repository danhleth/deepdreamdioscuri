from dioscuri.base.registry import Registry

from .ce_loss import ClassificationCELoss

CRITERION_REGISTRY = Registry('CRITERION')

CRITERION_REGISTRY.register(ClassificationCELoss)
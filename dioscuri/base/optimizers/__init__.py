from torch.optim import SGD, Adam 


from dioscuri.base.registry import Registry

from dioscuri.base.optimizers import lr_scheduler

OPTIMIZER_REGISTRY = Registry('OPTIMIZER')
OPTIMIZER_REGISTRY.register(Adam)
OPTIMIZER_REGISTRY.register(SGD)

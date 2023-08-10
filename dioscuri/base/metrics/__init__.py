from dioscuri.base.registry import Registry
from .metric_template import Metric
from .accuracy import ClassificationAccuracy

METRIC_REGISTRY = Registry('METRIC')

METRIC_REGISTRY.register(ClassificationAccuracy)
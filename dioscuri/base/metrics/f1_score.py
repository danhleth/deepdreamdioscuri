from typing import Any, Dict

from sklearn.metrics import f1_score

from dioscuri.base.metrics.metric_template import Metric


class ClassificationF1ScoreMetric(Metric):
    """
    F1 Score Metric (including macro, micro)
    """

    def __init__(self, average="weighted", label_type: str = "multiclass", **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.type = label_type
        self.reset()

    def update(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """
        Perform calculation based on prediction and targets
        """
        targets = batch["label"]
        outputs = outputs["logit"]

        self.preds += outputs.numpy().tolist()
        self.targets += targets.numpy().tolist()

    def value(self):
        score = f1_score(self.targets, self.preds, average=self.average)
        return score

    def reset(self):
        self.targets = []
        self.preds = []
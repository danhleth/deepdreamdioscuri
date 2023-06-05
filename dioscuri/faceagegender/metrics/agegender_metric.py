from typing import Any, List, Optional, Tuple, Dict 

import torch

from dioscuri.base.metrics import METRIC_REGISTRY
from dioscuri.base.metrics.metric_template import Metric

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

@METRIC_REGISTRY.register()
class AgeEstGenderAccuracy(Metric):
    """Age Estimation and Gender Accuracy
    """
    def __init__(self, ignore_index: Optional[Any] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reset()

    def update(self, output: torch.Tensor, batch: Dict[str, Any]):
        
        logit_age = output['logit_age'] if isinstance(output, Dict) else output[0]
        logit_gender = output['logit_gender'] if isinstance(output, Dict) else output[1]

        target_age = batch["label_age"] if isinstance(batch, Dict) else batch
        target_gender = batch["label_gender"] if isinstance(batch, Dict) else batch

        logit_age = logit_age.cpu().numpy()
        logit_gender = logit_gender.cpu().numpy()
        target_age = target_age.cpu().numpy()
        target_gender = target_gender.cpu().numpy()

        self.accuracy = accuracy_score(y_pred=logit_gender, y_true=target_gender)
        self.mae = mean_absolute_error(y_pred = logit_age, y_true=target_age)
        self.mse = mean_squared_error(y_pred = logit_age, y_true=target_age)
        

    def value(self):
        return {'accuracy': self.accuracy,
                'mae': self.mae,
                'mse': self.mse}

    def reset(self):
        self.accuracy = 0
        self.mae = 0
        self.mse = 0

    def summary(self):
        print(f"Accuracy: {self.accuracy} \t MSE: {self.mse} \t MAE: {self.mae}")
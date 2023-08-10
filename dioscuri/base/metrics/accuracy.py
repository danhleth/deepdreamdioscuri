from typing import Any, List, Optional, Tuple, Dict 

import torch
import numpy as np

from dioscuri.base.metrics.metric_template import Metric

class ClassificationAccuracy(Metric):
    """ Calculate accuracy metric for multi-class classification task
    """
    def __init__(self) -> None:
        super().__init__()
        self.reset()
    
    def update(self, output: torch.Tensor, batch: Dict[str, Any]):
        """ Update accuracy metric
            logit: (batch_size, vector_classes)
            target: (batch_size, one_hot_vector_classes)
        """
        logit = output['logit'] if isinstance(output, Dict) else output[1]
        target = batch["label"] if isinstance(batch, Dict) else batch
        logit = logit.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        

        logit = np.argmax(logit, axis=1)
        target = np.argmax(target, axis=1)


        correct = np.sum(logit == target)
        self.total_correct += correct
        self.total_count += len(target)
        
    
    def value(self):
        return (self.total_correct / self.total_count)

    def reset(self):
        self.total_correct = 0
        self.total_count = 0

    def summary(self):
        print(f"Accuracy: {(self.total_correct / self.total_count)}")
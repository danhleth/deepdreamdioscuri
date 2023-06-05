from typing import Any, Dict, List
from dioscuri.base.trainer import TRAINER_REGISTRY

import logging
import numpy as np
from tqdm.auto import tqdm as tqdm
import torch
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dioscuri.base.trainer.supervised_trainer import SupervisedTrainer
from dioscuri.base.metrics.metric_template import Metric
from dioscuri.utils.device import detach, move_to
from dioscuri.utils.meter import AverageValueMeter


from . import TRAINER_REGISTRY


@TRAINER_REGISTRY.register()
class AgeGenderTrainer(SupervisedTrainer):
    def __init__(self, 
                 cfg: Any,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 metrics: Dict[str, Metric],
                 model: Module,
                 scheduler,
                 optimizer: Optimizer,
                 device: device="cuda"):
        
        super().__init__(
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            metrics=metrics,
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            device=device
        )

    def train_epoch(self, epoch:int, dataloader: DataLoader) -> List:
        """Training epoch

        Args: 
            epoch(int)
            dataloader
        
        returns:
            float, float
        """
        running_loss = AverageValueMeter()
        total_loss = AverageValueMeter()
        for m in self.metric.values():
            m.reset()
        self.model.train()
        print("Training........")
        progress_bar = tqdm(dataloader) if self.verbose else dataloader
        for i, batch in enumerate(progress_bar):
            # 1: Load img_inputs and labels
            batch = move_to(batch, self.device)

            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            with autocast(enabled=self.cfg.fp16):
                # 3: Get network outputs
                # 4: Calculate the loss
                out_dict = self.model(batch)
            # 5: Calculate gradients
            self.scaler.scale(out_dict['loss']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # 6: Performing backpropagation
            with torch.no_grad():
                # 7: Update loss
                running_loss.add(out_dict['loss'].item())
                total_loss.add(out_dict['loss'].item())

                if (i + 1) % self.cfg.log_step == 0 or (i + 1) == len(dataloader):
                    self.tsboard.update_loss(
                        "train", running_loss.value(
                        )[0], epoch * len(dataloader) + i
                    )
                    running_loss.reset()

                # 8: Update metric
                outs = detach(out_dict)
                batch = detach(batch)
                for m in self.metric.values():
                    m.update(outs, batch)
                    
        self.save_result(outs, batch, stage="train")
        avg_loss = total_loss.value()[0]
        return avg_loss
        pass

    def fit(self):
        for epoch in range(self.cfg.nepochs):

            # Note learning rate
            for i, group in enumerate(self.optimizer.param_groups):
                self.tsboard.update_lr(i, group["lr"], epoch)

            self.epoch = epoch
            logging.info(f"\nEpoch {epoch:>3d}")
            logging.info("-----------------------------------")

            # 1: Training phase
            # 1.1 train
            avg_loss = self.train_epoch(
                epoch=epoch, dataloader=self.train_data)

            # 1.2 log result
            logging.info("+ Training result")
            logging.info(f"Loss: {avg_loss}")
            for m in self.metric.values():
                m.summary()

            # 2: Evalutation phase
            if (epoch + 1) % self.cfg.val_step == 0:
                with autocast(enabled=self.cfg.fp16):
                    # 2: Evaluating model
                    avg_loss = self.evaluate(epoch, dataloader=self.val_data)

                    logging.info("+ Evaluation result")
                    logging.info(f"Loss: {avg_loss}")

                    for m in self.metric.values():
                        m.summary()

                    # 3: Learning rate scheduling
                    self.scheduler.step(avg_loss)

                    # 4: Saving checkpoints
                    if not self.cfg.debug:
                        # Get latest val loss here
                        val_metric = {k: m.value()
                                      for k, m in self.metric.items()}
                        self.save_checkpoint(epoch, avg_loss, val_metric)
            logging.info("-----------------------------------")

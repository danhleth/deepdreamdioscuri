import logging
from typing import Callable, Dict, Optional
import yaml
from pathlib import Path

import torch
from torch.utils.data.dataset import random_split

from dioscuri.base.opt import Opts

from dioscuri.base.optimizers import OPTIMIZER_REGISTRY
from dioscuri.base.optimizers.lr_scheduler import SCHEDULER_REGISTRY
from dioscuri.base.transforms.alb import TRANSFORM_REGISTRY
from dioscuri.base.criterion import CRITERION_REGISTRY
from dioscuri.base.datasets import DATASET_REGISTRY
from dioscuri.base.trainer import TRAINER_REGISTRY
from dioscuri.base.models import MODEL_REGISTRY
from dioscuri.base.metrics import METRIC_REGISTRY
from dioscuri.base.test import evaluate
from dioscuri.base.models.wrapper import ModelWithLoss
from dioscuri.utils.getter import (get_dataloader, 
                               get_instance, 
                               get_instance_recursively)
from dioscuri.utils.loading import load_yaml


class BasePipeline:
    """docstring for Pipeline."""

    def __init__(
        self,
        opt: Opts,
        cfg_path: Optional[str] = None,
        transform_cfg_path: Optional[str] = None,
    ):  
        
        self.cfg_pipeline = opt["pipeline"]
        self.cfg_transform = opt["transform"]
        opt = opt["opt"]
        self.opt = opt

        self.device = opt["device"]
        print(self.device)
        torch.cuda.set_device(self.device)
        self.transform = None
        if self.cfg_transform is not None:
            self.transform = get_instance_recursively(
                self.cfg_transform, registry=TRANSFORM_REGISTRY
            )
        data = self.get_data(self.cfg_pipeline["data"], self.transform, return_dataset=False)
        self.train_dataloader,self.val_dataloader,self.train_dataset,self.val_dataset = data
        
        backbone = get_instance(self.cfg_pipeline["model"], registry=MODEL_REGISTRY).to(self.device)

        criterion = get_instance(self.cfg_pipeline["criterion"], registry=CRITERION_REGISTRY)
        
        self.model = ModelWithLoss(backbone, criterion)

    
        self.metric = {
            mcfg["name"]: get_instance(mcfg, registry=METRIC_REGISTRY)
            for mcfg in self.cfg_pipeline["metric"]
        }

        self.optimizer = get_instance(
            self.cfg_pipeline["optimizer"],
            registry=OPTIMIZER_REGISTRY,
            params=self.model.parameters(),
        )

        self.scheduler = get_instance(
            self.cfg_pipeline["scheduler"], registry=SCHEDULER_REGISTRY, optimizer=self.optimizer
        )

        self.trainer = get_instance(
            self.cfg_pipeline["trainer"],
            cfg=self.opt,
            train_data=self.train_dataloader,
            val_data=self.val_dataloader,
            scheduler=self.scheduler,
            model=self.model,
            metrics=self.metric,
            optimizer=self.optimizer,
            registry=TRAINER_REGISTRY,
            device=self.device
        )

        save_cfg = {}
        save_cfg["opt"] = opt
        save_cfg["pipeline"] = self.cfg_pipeline
        save_cfg['transform'] = self.cfg_transform
        save_cfg["opt"]["save_dir"] = str(save_cfg["opt"]["save_dir"])
        with open(
            self.trainer.save_dir / "checkpoints" / "config.yaml", "w"
        ) as outfile:
            yaml.dump(save_cfg, outfile, default_flow_style=False)
        self.logger = logging.getLogger()

    def sanitycheck(self):
        self.logger.info("Sanity checking before training")
        

    def fit(self):
        self.sanitycheck()
        self.trainer.fit()

    def evaluate(self):
        avg_loss, metric = evaluate(
            model=self.model,
            dataloader=self.val_dataloader,
            metric=self.metric,
            device=self.device,
            verbose=self.opt.verbose,
        )
        print("Evaluate result")
        print(f"Loss: {avg_loss}")
        for m in metric.values():
            m.summary()

    def get_data(
        self, cfg, transform: Optional[Dict[str, Callable]] = None, return_dataset=False
    ):
        def get_single_data(cfg, transform, stage: str = "train"):
            assert stage in cfg["dataset"].keys(), f"{stage} is not in dataset config"
            assert stage in cfg["loader"].keys(), f"{stage} is not in loader config"

            if transform is None:
                dataset = get_instance(cfg["dataset"][stage], registry=DATASET_REGISTRY)
            else:
                dataset = get_instance(
                    cfg["dataset"][stage],
                    registry=DATASET_REGISTRY,
                    transform=transform[stage],
                )
            dataloader = get_dataloader(cfg["loader"][stage], dataset, local_rank=self.device)
            return dataloader, dataset

        train_dataloader, train_dataset = None, None
        if ("train" in cfg["dataset"]) and ("train" in cfg["loader"]):
            train_dataloader, train_dataset = get_single_data(cfg, transform, "train")

        val_dataloader, val_dataset = None, None
        if ("val" in cfg["dataset"]) and ("val" in cfg["loader"]):
            val_dataloader, val_dataset = get_single_data(cfg, transform, "val")

        if train_dataloader is None and val_dataloader is None:
            dataset = get_instance(
                cfg["dataset"], registry=DATASET_REGISTRY, transform=None
            )
            train_sz, val_sz = cfg["splits"]["train"], cfg["splits"]["val"]
            train_sz = int(len(dataset) * train_sz)
            val_sz = len(dataset) - train_sz
            assert (
                val_sz > 0
            ), f"validation size must be greater than 0. val_sz = {val_sz}"
            train_dataset, val_dataset = random_split(dataset, [train_sz, val_sz])
            if transform is not None:
                train_dataset.dataset.transform = transform["train"]
                val_dataset.dataset.transform = transform["val"]
            train_dataloader = get_dataloader(cfg["loader"]["train"], train_dataset)
            val_dataloader = get_dataloader(cfg["loader"]["val"], val_dataset)

        return (train_dataloader, val_dataloader, train_dataset, val_dataset)


    def export_register(self):
        self.logger.info("Exporting register")
        register_save_dir = Path(__file__).parent.parent.parent/ "docs" / "register"
        register_save_dir.mkdir(exist_ok=True)
        MODEL_REGISTRY.export_to_file(register_save_dir / "model.txt")
        OPTIMIZER_REGISTRY.export_to_file(register_save_dir / "optimizer.txt")
        SCHEDULER_REGISTRY.export_to_file(register_save_dir / "scheduler.txt")
        CRITERION_REGISTRY.export_to_file(register_save_dir  / "criterion.txt")
        DATASET_REGISTRY.export_to_file(register_save_dir  / "dataset.txt")
        TRAINER_REGISTRY.export_to_file(register_save_dir  / "trainer.txt")
        METRIC_REGISTRY.export_to_file(register_save_dir  / "metric.txt")
        TRANSFORM_REGISTRY.export_to_file(register_save_dir  / "transform.txt")
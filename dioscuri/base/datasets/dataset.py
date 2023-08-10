import os
from typing import Iterable, List

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import pandas as pd


class ConcatDataset(data.ConcatDataset):
    """
    Concatenate dataset and do sampling randomly

    datasets: `Iterable[data.Dataset]`
        list of datasets
    """

    def __init__(self, datasets: Iterable[data.Dataset], **kwargs) -> None:
        super().__init__(datasets)

        # Workaround, not a good solution
        self.classnames = datasets[0].classnames
        self.collate_fn = datasets[0].collate_fn

    def __getattr__(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)

        if hasattr(self.datasets[0], attr):
            return getattr(self.datasets[0], attr)

        raise AttributeError


class ChainDataset(data.ConcatDataset):
    """
    Chain dataset and do sampling iteratively

    datasets: `Iterable[data.Dataset]`
        list of datasets
    """

    def __init__(self, datasets: Iterable[data.Dataset], **kwargs) -> None:
        super().__init__(datasets)

        # Workaround, not a good solution
        self.classnames = datasets[0].classnames
        self.collate_fn = datasets[0].collate_fn


class ImageDataset(data.Dataset):
    """
    Dataset contains folder of images

    image_dir: `str`
        path to folder of images
    txt_classnames: `str`
        path to .txt file contains classnames
    transform: `List`
        list of transformation
    """

    def __init__(
        self,
        image_dir: str,
        txt_classnames: str = None,
        transform: List = None,
        **kwargs
    ):
        super().__init__()
        self.image_dir = image_dir
        self.txt_classnames = txt_classnames
        self.transform = transform
        self.load_data()

    def load_data(self):
        """
        Load filepaths into memory
        """
        if self.txt_classnames:
            with open(self.txt_classnames, "r") as f:
                self.classnames = f.read().splitlines()
        self.fns = []
        image_names = os.listdir(self.image_dir)
        for image_name in image_names:
            self.fns.append(image_name)

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_name = self.fns[index]
        image_path = os.path.join(self.image_dir, image_name)
        im = Image.open(image_path).convert("RGB")
        width, height = im.width, im.height

        if self.transform is not None:
            try:
                im = self.transform(im)
            except:
                im = self.transform(image=np.array(im) / 255.0)["image"]

        return {
            "input": im.float(),
            "img_name": image_name,
            "ori_size": [width, height],
        }

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch: List):
        imgs = torch.stack([s["input"] for s in batch])
        img_names = [s["img_name"] for s in batch]
        ori_sizes = [s["ori_size"] for s in batch]

        return {"inputs": imgs, "img_names": img_names, "ori_sizes": ori_sizes}


class Image2DDataset(data.Dataset):
    """
        Dataset contains folder of images

        root_dir: `str`
            path to folder of images
        annotation: `str`
            path to annotation file of dataset
            default: image_path, label_id
        transform:
            list of transformation
    """
    def __init__(self, root_dir, annotation, transform=None):
        self.root_dir = root_dir
        self.annotation = annotation
        self.transform = transform
        self.df = self.load_annotation()
    
    def load_annotation(self):
        df = pd.read_csv(self.annotation)
        return df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = cv2.imread(img_name)
        label = self.df.iloc[idx, 1]
        image /= 255.0
        if self.transform:
            try:
                image = self.transform(image)
            except:
                image = self.transform(image=np.array(image))["image"]
        return {"input": image.float(), 
                "label": label, 
                "img_name": img_name, 
                "ori_size": [image.shape[0], image.shape[1]]}

import os
from typing import List, Optional, Tuple, Dict 

import torch
import pandas as pd
import numpy as np
from PIL import Image
from dioscuri.base.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class AFADDataset(torch.utils.data.Dataset):
    r"""AFADDataset multi-classes face dataset
    """
    def __init__(self, root_dir, annotation_file, transform):
        super(AFADDataset,self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.exists(annotation_file):
            annotation_file = os.path.join(root_dir, annotation_file)
        self.df = pd.read_csv(annotation_file)[:124]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int) -> Dict:
        items = self.df.iloc[idx]
        image = items['image_name']
        image = os.path.join(self.root_dir, image)
        image =  Image.open(image).convert('RGB')

        age = items['age']
        age = np.float32(age)

        gender = items['gender']
        gender = np.float32(gender)
        
        if self.transform:
            image = self.transform(image)

        sample ={'input': image, 'label_age': age, 'label_gender': gender}
        return sample
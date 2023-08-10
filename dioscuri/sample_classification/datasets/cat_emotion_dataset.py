from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset

import cv2
import pandas as pd

class CATEMOTIONDATASET(Dataset):
    """ Dataset contains folder of images
        source: https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset
    """
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = Path(root_dir)
        self.annotation_file = Path(annotation_file)
        self.transform = transform
        self.df = pd.read_csv(self.annotation_file)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
            self.df includes:
                - image_path
                - label_id
                - label_name
        """
        image_path = self.root_dir / self.df.iloc[idx, 0]
        label_id = self.df.iloc[idx, 1]
        label_name = self.df.iloc[idx, 2]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image /= 255.0

        if self.transform:
            try:
                image = self.transform(image)
            except:
                image = self.transform(image)["image"]
        
        return {"input": image, 
                "label_id": label_id}
        
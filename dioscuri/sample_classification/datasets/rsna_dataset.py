from typing import List, Tuple
from pathlib import Path
from torch.utils.data import Dataset

import cv2
import pandas as pd
import numpy as np
import pydicom



class RSNADATASET(Dataset):
    """
        Dataset contains folder of images
        source: https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection

        train.csv Target labels for the train set. Note that patients labeled healthy may still have other medical issues, such as cancer or broken bones, that don't happen to be covered by the competition labels.

        patient_id - A unique ID code for each patient.
        [bowel/extravasation]_[healthy/injury] - The two injury types with binary targets.
        [kidney/liver/spleen]_[healthy/low/high] - The three injury types with three target levels.
        any_injury - Whether the patient had any injury at all.
        [train/test]_images/[patient_id]/[series_id]/[image_instance_number].dcm The CT scan data, in DICOM format. Scans from dozens of different CT machines have been reprocessed to use the run length encoded lossless compression format but retain other differences such as the number of bits per pixel, pixel range, and pixel representation. Expect to see roughly 1,100 patients in the test set.

        [train/test]_series_meta.csv Each patient may have been scanned once or twice. Each scan contains a series of images.

        patient_id - A unique ID code for each patient.
        series_id - A unique ID code for each scan.
        aortic_hu - The volume of the aorta in hounsfield units. This acts as a reliable proxy for when the scan was. For a multiphasic CT scan, the higher value indicates the late arterial phase.
        incomplete_organ - True if one or more organs wasn't fully covered by the scan. This label is only provided for the train set.
        sample_submission.csv A valid sample submission. Only the first few rows are available for download.

        image_level_labels.csv Train only. Identifies specific images that contain either bowel or extravasation injuries.

        patient_id - A unique ID code for each patient.
        series_id - A unique ID code for each scan.
        instance_number - The image number within the scan. The lowest instance number for many series is above zero as the original scans were cropped to the abdomen.
        injury_name - The type of injury visible in the frame.
    """
    def __init__(self, root_dir, annotation_file):
        self.root_dir = Path(root_dir)
        self.images = sorted(list(self.root_dir.glob('**/**/*.dcm')))
        self.df = pd.read_csv(annotation_file)
    
    def __len__(self):
        return len(self.images)

    def get_patent_id_from_image_path(self, image_path: Path) -> str:
        return str(image_path).split('/')[-3]
    
    def __getitem__(self, index):
        image = pydicom.dcmread(self.images[index])
        image = image.pixel_array
        patent_id = self.get_patent_id_from_image_path(self.images[index])
        label = self.df[self.df['patient_id'] == int(patent_id)].values[0][1:]
        return {"input": image, "label": label}
import random
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

class SegDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None
    ):
        root_path = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        
        self.images = sorted((root_path / "images").iterdir())
        self.masks = sorted((root_path / "segmentation").iterdir())

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        image, mask = self.joint_transform(image, mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
    
CITYSCAPES_LABEL_MAP = np.full(256, 255, dtype=np.uint8)
CITYSCAPES_LABEL_MAP[7]  = 0  # road
CITYSCAPES_LABEL_MAP[8]  = 1  # sidewalk
CITYSCAPES_LABEL_MAP[11] = 2  # building
CITYSCAPES_LABEL_MAP[12] = 3  # wall
CITYSCAPES_LABEL_MAP[13] = 4  # fence
CITYSCAPES_LABEL_MAP[17] = 5  # pole
CITYSCAPES_LABEL_MAP[19] = 6  # traffic light
CITYSCAPES_LABEL_MAP[20] = 7  # traffic sign
CITYSCAPES_LABEL_MAP[21] = 8  # vegetation
CITYSCAPES_LABEL_MAP[22] = 9  # terrain
CITYSCAPES_LABEL_MAP[23] = 10  # sky
CITYSCAPES_LABEL_MAP[24] = 11  # person
CITYSCAPES_LABEL_MAP[25] = 12  # rider
CITYSCAPES_LABEL_MAP[26] = 13  # car
CITYSCAPES_LABEL_MAP[27] = 14  # truck
CITYSCAPES_LABEL_MAP[28] = 15  # bus
CITYSCAPES_LABEL_MAP[31] = 16  # train
CITYSCAPES_LABEL_MAP[32] = 17  # motorcycle
CITYSCAPES_LABEL_MAP[33] = 18  # bicycle
    
def load_dataset(dataset: str, root_dir: str, train: bool = True):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
            std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0],
        )
    ])

    if dataset == "cityscapes":
        target_transform = lambda x: torch.from_numpy(
            CITYSCAPES_LABEL_MAP[np.array(x, dtype=np.uint8)].astype(np.int64)
        )
    else:
        target_transform = lambda x: torch.from_numpy(np.array(x, dtype=np.int64))

    if train:
        color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

        def joint_transform(image, mask):
            crop = (512, 1024)

            i, j, h, w = T.RandomCrop.get_params(image, crop)

            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            image = color_jitter(image)

            return image, mask
    else:
        def joint_transform(image, mask):
            if dataset == "trafficsynth":
                size = (512, 1024)
            elif dataset == "cityscapes":
                size = (1024, 2048)

            image = TF.resize(image, size, interpolation=Image.BILINEAR)
            mask = TF.resize(mask, size, interpolation=Image.NEAREST)

            return image, mask
        
    return SegDataset(root_dir, transform, target_transform, joint_transform)

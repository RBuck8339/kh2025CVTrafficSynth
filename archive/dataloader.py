"""
Mapping Dataset:
Carla: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
TrafficCAM: Truck, Bus, Motor Bike, bike, Pedestrian, LMV

Not in Carla (NC): Tractor(NC), E-rickshaw(NC), LCV(NC), Auto(NC)  

Carla to TrafficCam (the labels we will save):
Truck --> Truck
Bus --> Bus
Motorcycle --> Motor Bike
Bicycle --> bike
Pedestrian --> Pedestrian
Car --> LMV
"""
import torch
import os
import numpy as np
from enum import Enum
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class Label(Enum):
    IGNORE = 0
    TRUCK = 1
    BUS = 2
    MOTORCYCLE = 3
    BICYCLE = 4
    PEDESTRIAN = 5
    CAR = 6

# RGB to Label
CARLA_TO_LABEL = {
    (0, 0, 70): Label.TRUCK,
    (0, 60, 100): Label.BUS,
    (0, 0, 230): Label.MOTORCYCLE,
    (119, 11, 32): Label.BICYCLE,
    (220, 20, 60): Label.PEDESTRIAN,
    (0, 0, 142): Label.CAR
}

# string to label
TRAFFIC_CAM_TO_LABEL = {
    "Truck": Label.TRUCK,
    "Bus": Label.BUS,
    "MotorBike": Label.MOTORCYCLE,
    "Bike": Label.BICYCLE,
    "Pedestrian": Label.PEDESTRIAN,
    "LMV": Label.CAR
}

class CarlaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # sort files via frame number
        sort_key = lambda file: int(file[len("frame_"):-1*len(".png")])
        self.images = sorted(os.listdir(image_dir), key=sort_key)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # if data is corrupted
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"[WARNING] Corrupt image skipped: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        mask_rgb = Image.open(mask_path).convert("RGB")
        mask = self.rgb_to_mask(np.array(mask_rgb))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.

        return image.float(), torch.from_numpy(mask).long()

    def rgb_to_mask(self, rgb):
        h, w, _ = rgb.shape
        mask = np.full((h, w), Label.IGNORE.value, dtype=np.uint8)

        for color, label in CARLA_TO_LABEL.items():
            matches = np.all(rgb == color, axis=-1)
            mask[matches] = label.value

        return mask

# Traffic Cam

import json
import cv2

class TrafficCamDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform


        # filter out json files from images
        self.images = list(filter(lambda file_name: file_name.endswith(".jpg"), os.listdir(image_dir)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.image_dir, img_file)
        json_path = os.path.join(self.json_dir, img_file.replace(".jpg", ".json"))

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"[WARNING] Corrupt image skipped: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        width, height = image.size

        mask = self.json_to_mask(json_path, height, width)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2,0,1) / 255.

        return image.float(), torch.from_numpy(mask).long()

    def json_to_mask(self, path, height, width):
        """
        translates json into a mask image for validation
        """
        mask = np.full((height, width), Label.IGNORE.value, dtype=np.uint8)

        with open(path) as f:
            data = json.load(f)

        for shape in data["shapes"]:
            label = shape["label"]

            if label not in TRAFFIC_CAM_TO_LABEL:
                continue

            points = np.array(shape["points"], dtype=np.int32)
            class_id = TRAFFIC_CAM_TO_LABEL[label].value

            cv2.fillPoly(mask, [points], class_id)
        return mask
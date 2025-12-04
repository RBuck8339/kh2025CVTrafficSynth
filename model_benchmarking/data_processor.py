import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro
from tqdm import tqdm

segmentation_map = {
    "Roads": (0, (128, 64, 128)),
    "SideWalks": (1, (244, 35, 232)),
    "Building": (2, (70, 70, 70)),
    "Wall": (3, (102, 102, 156)),
    "Fence": (4, (190, 153, 153)),
    "Pole": (5, (153, 153, 153)),
    "TrafficLight": (6, (250, 170, 30)),
    "TrafficSign": (7, (220, 220, 0)),
    "Vegetation": (8, (107, 142, 35)),
    "Terrain": (9, (152, 251, 152)),
    "Sky": (10, (70, 130, 180)),
    "Pedestrian": (11, (220, 20, 60)),
    "Rider": (12, (255, 0, 0)),
    "Car": (13, (0, 0, 142)),
    "Truck": (14, (0, 0, 70)),
    "Bus": (15, (0, 60, 100)),
    "Train": (16, (0, 80, 100)),
    "Motorcycle": (17, (0, 0, 230)),
    "Bicycle": (18, (119, 11, 32)),
    "Static": (255, (110, 190, 160)),
    "Dynamic": (255, (170, 120, 50)),
    "Other": (255, (55, 90, 80)),
    "Water": (255, (45, 60, 150)),
    "RoadLine": (255, (157, 234, 50)),
    "Ground": (255, (81, 0, 81)),
    "Bridge": (255, (150, 100, 100)),
    "RailTrack": (255, (230, 150, 140)),
    "GuardRail": (255, (180, 165, 180)),
    "Void": (255, (0, 0, 0)),
}

def trafficsynth(dataset_dir: str, out_dir: str):
    dataset_dir_path = Path(dataset_dir)
    out_dir_path = Path(out_dir)

    img_dir = out_dir_path / "images"
    seg_dir = out_dir_path / "segmentation"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    frame_count = 0

    for camera in tqdm(dataset_dir_path.iterdir()):
        raw_dir = camera / "raw"
        semantic_dir = camera / "semantic"
        files = sorted(os.listdir(raw_dir), key=lambda name: int(name.split("_")[1].split(".")[0]))

        for file in files:
            raw_frame = cv2.imread(raw_dir / file)
            semantic_frame = cv2.imread(semantic_dir / file)

            H, W = raw_frame.shape[:2]

            out = np.full((H, W), 255, dtype=np.uint8)

            for label, (idx, color) in segmentation_map.items():
                mask = np.all(semantic_frame == color[::-1], axis=2)
                out[mask] = idx

            new_file = f"{frame_count:06d}.png"

            shutil.copy(raw_dir / file, img_dir / new_file)
            cv2.imwrite(seg_dir / new_file, out)
            frame_count += 1

def cityscapes(raw_dir: str, semantic_dir: str, out_dir: str):
    raw_dir_path = Path(raw_dir)
    semantic_dir_path = Path(semantic_dir)
    out_dir_path = Path(out_dir)

    img_dir = out_dir_path / "images"
    seg_dir = out_dir_path / "segmentation"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    frame_count = 0

    for city in tqdm(os.listdir(raw_dir_path)):
        raw_dir = raw_dir_path / city
        semantic_dir = semantic_dir_path / city
        files = sorted(["_".join(x.split("_")[:3]) for x in os.listdir(raw_dir)])

        for file in files:
            new_file = f"{frame_count:06d}.png"

            shutil.copy(raw_dir / f"{file}_leftImg8bit.png", img_dir / new_file)
            shutil.copy(semantic_dir / f"{file}_gtFine_labelIds.png", seg_dir / new_file)

            frame_count += 1

def main(dataset: str, dataset_dir: str | None = None, raw_dir: str | None = None, semantic_dir: str | None = None, out_dir: str | None = None):
    if dataset == "trafficsynth":
        trafficsynth(dataset_dir, out_dir)
    elif dataset == "cityscapes":
        cityscapes(raw_dir, semantic_dir, out_dir)

if __name__ == "__main__":
    tyro.cli(main)

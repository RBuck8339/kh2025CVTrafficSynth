import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader 

import dataloader
from models import DDRNet_23 as DDRNet

from PIL import Image
from torchmetrics import JaccardIndex
import cv2

# config
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IOU
iou = JaccardIndex(task="multiclass", num_classes=7).to(DEVICE)

# transform
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

# Synthetic data
dataset_carla = dataloader.CarlaDataset(
    image_dir="data/carla_captures/Foggy/camera0/raw",
    mask_dir="data/carla_captures/Foggy/camera0/semantic",
    transform=transform,
)

# real data
TrafficCAM_DIR = "data/TrafficCAM-fully-annotated/UCF_1651231637.4264545"
dataset_TrafficCAM = dataloader.TrafficCamDataset(
    image_dir=TrafficCAM_DIR,
    json_dir=TrafficCAM_DIR,
    transform=transform,
)

# loaders
train_loader = DataLoader(
    dataset_carla,
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
)

test_loader = DataLoader(
    dataset_TrafficCAM, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
)

# DDRNet model
model = DDRNet.DualResNet(DDRNet.BasicBlock, [2, 2, 2, 2], num_classes=6, planes=64, spp_planes=128, head_planes=128, augment=False).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    running_iou = 0

    # loading bar
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}")

    # train logic
    for images, masks in pbar:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        # resize
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
    
        # iou = mean_iou(preds, masks)
        iou_score = iou(preds, masks)

        running_loss += loss.item()
        running_iou += iou_score 

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "IoU": f"{iou_score:.3f}"
        })

    print(f"Epoch {epoch+1}: "
          f"Loss={running_loss/len(train_loader):.4f}, "
          f"IoU={running_iou/len(train_loader):.3f}")
    
    # |====== test logic =====|
    model.eval()
    test_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            outputs = model(images)

            # resize model
            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

            preds = outputs.argmax(1)
            test_iou += iou(preds, masks)

    print(f"TEST (TrafficCAM): IoU={test_iou/len(test_loader):.3f}")
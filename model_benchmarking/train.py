from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import load_dataset
from tqdm import tqdm
import numpy as np
import tyro

from utils.DDRNet_23 import load_ddrnet, load_ddrnet_slim

def main(
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-2,
    model: str = "DDRNet-23",
    checkpoint_path: Optional[str] = None,
    train_dataset: str = "cityscapes",
    train_path: str = "data/cityscapes_train",
    val_dataset: str = "cityscapes",
    val_path: str = "data/cityscapes_val"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_weight = torch.tensor([
        0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
        1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
        1.0507
    ], dtype=torch.float32, device=device)

    train_data = load_dataset(dataset=train_dataset, root_dir=train_path, train=True)
    val_data = load_dataset(dataset=val_dataset, root_dir=val_path, train=False)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)

    if model == "DDRNet-23":
        model = load_ddrnet()
    elif model == "DDRNet-23-Slim":
        model = load_ddrnet_slim()
    else:
        raise Exception("Unsupported model")

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu') 
        model.load_state_dict(checkpoint, strict = False)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=255).to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    def poly_lr_scheduler(optimizer, base_lr, iter, max_iter, power):
        lr = base_lr * (1 - iter / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    train_loss_values, train_miou_values = [], []
    val_loss_values, val_miou_values = [], []

    global_iter = 0
    max_iter = epochs * len(train_loader)

    for epoch in range(epochs):
        model.train()
        model.augment = True

        losses = []
        conf_mat = torch.zeros(19, 19, dtype=torch.long)

        for images, masks in tqdm(train_loader, leave=False):
            poly_lr_scheduler(optimizer, learning_rate, global_iter, max_iter, power=0.9)
            global_iter += 1

            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            main_logits, aux_logits = model(images)

            loss_main = criterion(main_logits, masks)
            loss_aux = criterion(aux_logits, masks)
            loss = loss_main + 0.4 * loss_aux

            preds = main_logits.argmax(dim=1)

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

            p = preds.view(-1)
            t = masks.view(-1)

            p = p[t != 255]
            t = t[t != 255]

            p = p[(t >= 0) & (t < 19)]
            t = t[(t >= 0) & (t < 19)]

            idx = t * 19 + p
            hist = torch.bincount(idx, minlength=19 ** 2).reshape(19, 19)
            conf_mat += hist.cpu()

        train_loss = float(np.mean(losses))

        tp = conf_mat.diag()
        fn = conf_mat.sum(dim=1) - tp
        fp = conf_mat.sum(dim=0) - tp
        union = tp + fp + fn

        iou = tp.float() / union.clamp(min=1).float()
        train_miou = iou[union > 0].mean().item()

        model.eval()
        model.augment = False

        losses = []
        conf_mat = torch.zeros(19, 19, dtype=torch.long)

        with torch.no_grad():
            for images, masks in tqdm(val_loader, leave=False):
                images, masks = images.to(device), masks.to(device)

                logits = model(images)

                loss = criterion(logits, masks)

                preds = logits.argmax(dim=1)

                losses.append(loss.item())

                p = preds.view(-1)
                t = masks.view(-1)

                p = p[t != 255]
                t = t[t != 255]

                p = p[(t >= 0) & (t < 19)]
                t = t[(t >= 0) & (t < 19)]

                idx = t * 19 + p
                hist = torch.bincount(idx, minlength=19 ** 2).reshape(19, 19)
                conf_mat += hist.cpu()

        val_loss = float(np.mean(losses))
        
        tp = conf_mat.diag()
        fn = conf_mat.sum(dim=1) - tp
        fp = conf_mat.sum(dim=0) - tp
        union = tp + fp + fn

        iou = tp.float() / union.clamp(min=1).float()
        val_miou = iou[union > 0].mean().item()

        train_loss_values.append(train_loss)
        train_miou_values.append(train_miou)
        val_loss_values.append(val_loss)
        val_miou_values.append(val_miou)

        print(f"Epoch {epoch+1}: Train_Loss={train_loss:.4f}, Train_mIoU={train_miou*100:.2f}, Val_Loss={val_loss:.4f}, Val_mIoU={val_miou*100:.2f}")

if __name__ == "__main__":
    tyro.cli(main)

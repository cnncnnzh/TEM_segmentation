import argparse
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import models
from xin_dataset import Xin_DataLoader

dir_img = Path('/home/birolt/zhu00336/Vasp_Cal/ML/Pytorch-UNet/TEM-ImageNet-v1.3/image/')
dir_mask = Path('/home/birolt/zhu00336/Vasp_Cal/ML/Pytorch-UNet/TEM-ImageNet-v1.3/circularMask/')
dir_checkpoint = Path('./checkpoints/')


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        amp: bool = False,
):
    dataset = Xin_DataLoader(dir_img, dir_mask)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    train_losses = []
    val_scores = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            images, true_masks = batch['image'], batch['mask']

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            masks_pred = model(images)
            loss = criterion(masks_pred.squeeze(1), true_masks.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss)

        model.eval()
        num_val_batches = len(val_loader)
        dice_score = 0

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in val_loader:
                image, mask_true = batch['image'], batch['mask']

                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                mask_pred = model(image)

                mask_pred = mask_pred.squeeze(1)
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

        val_score = dice_score / max(num_val_batches, 1)
        val_scores.append(float(val_score))

        print(f'Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss:.4f}')
        print(f'Epoch [{epoch}/{epochs}], Validation Score: {val_score:.4f}')

        if save_checkpoint:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_losses.png')

    plt.figure()
    plt.plot(range(1, epochs + 1), val_scores, label='val_scores')
    plt.xlabel('Epoch')
    plt.ylabel('score')
    plt.legend()
    plt.savefig('validation_score.png')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = input(
        "Please select a model: 1. unet_2_layer, 2. unet_3_layer, 3. unet_4_layer, 4. unet_cnn, 5. unet_wo_skip")
    if model_name == '1':
        model = models.UNet2Layer(1, args.classes)
    elif model_name == '2':
        model = models.UNet3Layer(1, args.classes)
    elif model_name == '3':
        model = models.UNet4Layer(1, args.classes)
    elif model_name == '4':
        model = models.UNetCNN(1, args.classes)
    elif model_name == '5':
        model = models.UNetWOSkip(1, args.classes)
    else:
        raise ValueError("Please enter a number between 1-5")

    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)

    train_model(
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_percent=args.val / 100,
        save_checkpoint=True,
        amp=False
    )

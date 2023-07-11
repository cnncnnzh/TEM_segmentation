import os
import numpy as np
import random
import glob
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, ColorJitter


def random_crop(img, mask_ori, crop_size, base_size=256):
    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask_ori = mask_ori.transpose(Image.FLIP_LEFT_RIGHT)

    # random scale (short edge)
    short_size = random.randint(int(base_size * 0.5), int(base_size))
    w, h = img.size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask_ori = mask_ori.resize((ow, oh), Image.NEAREST)
    crop_h, crop_w = crop_size
    if (oh < crop_h) or (ow < crop_w):
        padh = crop_h - oh if oh < crop_h else 0
        padw = crop_w - ow if ow < crop_w else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask_ori = ImageOps.expand(mask_ori, border=(0, 0, padw, padh), fill=0)

    # random crop crop_size
    w, h = img.size
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    mask_ori = mask_ori.crop((x1, y1, x1 + crop_w, y1 + crop_h))

    return img, mask_ori


def center_crop(img, mask_ori, crop_size):
    crop_h, crop_w = crop_size
    w, h = img.size
    if w * crop_h > crop_w * h:
        oh = crop_h
        ow = int(1.0 * w * oh / h)
    else:
        ow = crop_w
        oh = int(1.0 * h * ow / w)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask_ori = mask_ori.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - crop_w) / 2.))
    y1 = int(round((h - crop_h) / 2.))
    img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    mask_ori = mask_ori.crop((x1, y1, x1 + crop_w, y1 + crop_h))

    return img, mask_ori


class Xin_DataLoader(Dataset):
    def __init__(self, root_image, root_gt, num_class=1, crop_size=None, base_size=256, is_train=True):
        self.images_root = root_image
        self.gt_root = root_gt
        self.mask_values = [0, 1]

        self.image_path_list = sorted(glob.glob(os.path.join(root_image, '*.png')))
        self.gt_path_list = sorted(glob.glob(os.path.join(root_gt, '*.png')))

        print(f'Training on {len(self.image_path_list)} images')

        self.base_size = base_size
        self.num_class = num_class
        self.crop_size = crop_size

        # data augmentation
        if is_train:
            self.input_transform = Compose(
                [ColorJitter(brightness=0.05, contrast=0.3, saturation=0.4, hue=0.08), ToTensor()])
            self.crop_function = random_crop
        else:
            self.input_transform = Compose([ToTensor()])
            self.crop_function = center_crop

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        now_gt_path = self.gt_path_list[index]

        with open(image_path, 'rb') as f:
            # open image and convert it to single channel. 
            image = Image.open(f).convert('L')

        with open(now_gt_path, 'rb') as f:
            # open ground-truth and convert it to single channel. 
            label_ori = Image.open(f).convert('L')

        if self.crop_size is not None:
            image, label_ori = self.crop_function(image, label_ori, self.crop_size, self.base_size)

        label_ori = np.array(label_ori)

        w, h = image.size
        mask_ori = np.zeros((h, w))  # (h, w, c) -> (c, h, w)
        mask_ori[label_ori == 255] = 1
        label = torch.as_tensor(mask_ori)

        image = self.input_transform(image)

        return {
            'image': image,
            'mask': label,
        }

    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    train_image_root = '/Users/xincoder/Documents/code/kexin/TEM-ImageNet-v1.3/image/'
    train_gt_root = '/Users/xincoder/Documents/code/kexin/TEM-ImageNet-v1.3/circularMask/'
    data = Xin_DataLoader(train_image_root, train_gt_root, crop_size=(256, 256), base_size=256)

    ####################################
    # for visualization
    import cv2
    import os

    save_folder = 'xin_vis'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for ind, (image, label) in enumerate(data, start=1):
        print(image.shape, label.shape, label.min(), label.max(), image.min(), image.max(), )
        save_image_path = os.path.join(save_folder, '{:05d}_image.jpg'.format(ind))
        save_mask_path = os.path.join(save_folder, '{:05d}_mask.jpg'.format(ind))
        cv2.imwrite(save_image_path, (image[0].cpu().numpy() * 255).astype(np.uint8))
        cv2.imwrite(save_mask_path, (label[0] * 255).astype(np.uint8))

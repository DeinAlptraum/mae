# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from bisect import bisect

from torchvision import datasets, transforms
from torch import cat as tcat
from PIL import Image

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def has_masks(path):
    if "empties" not in has_masks.__dict__:
        root = "/".join(path.split("/")[:-2])
        with open(root + "/emptyfiles.txt", "r") as f:
            has_masks.empties = set([entry.strip() for entry in f.readlines()])
    example = path.split("/")[-1]
    return example not in has_masks.empties

class MaskImageDataset(datasets.ImageFolder):
    def __init__(
        self,
        img_root: str,
        mask_root: str,
        coverage_ratio: float,
        common_transform = None,
        image_transform = None,
        **kwargs,
    ):
        super().__init__(img_root, is_valid_file=has_masks, **kwargs)
        self.common_transform = common_transform
        self.mask_root = mask_root
        self.coverage_ratio = coverage_ratio
        self.image_transform = image_transform
        self.toTensor = transforms.ToTensor()

    def closest_match(self, path):
        with open(path+"/table.txt", 'r') as f:
            masks = [entry.strip().split() for entry in f.readlines()]
        masks = [(float(entry[0]), entry[1]) for entry in masks]
        if self.coverage_ratio >= masks[-1][0]:
            return masks[-1][1]
        res = bisect(masks, (self.coverage_ratio, ''))
        if abs(masks[res-1][0] - self.coverage_ratio) < abs(masks[res][0] - self.coverage_ratio):
            return masks[res-1][1]
        return masks[res][1]

    def pil_loader(self, path: str, conversion_code="RGB", mask=False) -> Image.Image:
        if mask:
            path += "/" + self.closest_match(path)
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(conversion_code)

    def __getitem__(self, index: int) -> tuple:
        path, target = self.samples[index]

        img = self.pil_loader(path)
        example = "/".join(path.split("/")[-2:])
        
        mask_path = self.mask_root + "/" + example[:-5]
        mask = self.pil_loader(mask_path, 'L', True)

        img = self.image_transform(self.toTensor(img))
        mask = self.toTensor(mask)

        if img.shape == (3, mask.shape[2], mask.shape[1]):
            mask = mask.swapaxes(1,2)
        combined = tcat((img,mask))
        combined = self.common_transform(combined)
        # img, mask = combined[:3], combined[3:]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return combined, target


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

from . import transforms as T


class DetectionPresetTrain:
    def __init__(self, *, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0), crop_size):
        if data_augmentation == "bonecell":
            self.transforms = A.Compose([
                A.RandomCrop(width=crop_size, height=crop_size),
                A.Rotate(limit=180),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(saturation=(1 - 0.2, 1 + 0.2), hue=(-0.05, 0.05),
                              brightness=(1 - 0.1, 1 + 0.1), contrast=(1, 1), p=0.5),
                ToTensorV2(),
                T.ConvertImageDtype(torch.float),
            ],
                bbox_params=A.BboxParams(format='coco'))


        elif data_augmentation == "hflip":
            self.transforms = T.Compose(
                [
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "lsj":
            self.transforms = T.Compose(
                [
                    T.ScaleJitter(target_size=(1024, 1024)),
                    T.FixedSizeCrop(size=(1024, 1024), fill=mean),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "ssd":
            self.transforms = T.Compose(
                [
                    T.RandomPhotometricDistort(),
                    T.RandomZoomOut(fill=list(mean)),
                    T.RandomIoUCrop(),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "ssdlite":
            self.transforms = T.Compose(
                [
                    T.RandomIoUCrop(),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float),
                ]
            )
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self, crop_size):
        self.transforms = A.Compose(
            [
                A.RandomCrop(width=crop_size, height=crop_size),
                ToTensorV2(),
                T.ConvertImageDtype(torch.float),
            ],
            bbox_params=A.BboxParams(format='coco'))

    def __call__(self, img, target):
        return self.transforms(img, target)

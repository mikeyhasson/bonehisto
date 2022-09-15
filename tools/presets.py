import torch
from . import transforms as T


class DetectionPresetTrain:
    def __init__(self, *, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0), crop_size):
        if data_augmentation == "bonecell":
            self.transforms = T.Compose([
                T.FixedSizeCrop((crop_size,crop_size)),
                T.Rotate(limit=180),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomPhotometricDistort(saturation=(1 - 0.2, 1 + 0.2), hue=(-0.05, 0.05),
                                    brightness=(1 - 0.1, 1 + 0.1), contrast=(1, 1), p=0.25),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ])

        elif data_augmentation == "hflip":
            self.transforms = T.Compose(
                [
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
        self.transforms = T.Compose(
            [
                T.RandomCrop(width=crop_size, height=crop_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)

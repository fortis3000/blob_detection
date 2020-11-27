"""
Bulk dataset module
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Dataset:
    """
    Bulk dataset Dataset.
    Reads images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    Example:
        dataset = Dataset(image_folder, masks+folder, classes=['bulk'])
    """

    CLASSES = ["bulk"]

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        classes=None,
        augmentations=None,
        preprocessing=None,
    ):

        self.ids = os.listdir(images_dir)
        self.images_fps = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_fps = [
            os.path.join(masks_dir, mask_id) for mask_id in self.ids
        ]

        self.class_values = [self.CLASSES.index(c.lower()) + 1 for c in classes]

        self.augmentations = augmentations
        self.preprocessing = preprocessing

    def __getitem__(self, item):

        # read data
        image = cv2.imread(self.images_fps[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[item], 0)

        assert mask is not None, (
            f"Mask can't be readed,"
            f" mask: {self.masks_fps[item]},"
            f" image: {self.images_fps[item]}"
        )

        mask //= 255

        # extract class from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentations:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


def visualize(save_name: str = None, **images):
    """Plots images in one row"""
    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())

        # do not work with colab
        if image.shape[-1] == 1:
            plt.imshow(image.squeeze(-1))
        else:
            plt.imshow(image)

        if save_name:
            plt.savefig(save_name)

    plt.close()


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


if __name__ == "__main__":
    data_path = os.path.join("data", "raw")

    dataset = Dataset(
        images_dir=os.path.join(data_path, "val"),
        masks_dir=os.path.join(data_path, "val_masks"),
        classes=["bulk"],
        augmentations=None,
        preprocessing=None,
    )

    for i in range(20):
        image, mask, _ = dataset[i]

        print(image.shape)
        print(mask.shape)

        visualize(image=image, bulk_mask=mask)

        print(_)

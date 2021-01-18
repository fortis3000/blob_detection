"""
Module to find optimal threshold for binary mask prediction threshold.
"""
import os
import cv2
import numpy as np

from segmentation_models.losses import DiceLoss
from segmentation_models import Unet
from config import BACKBONE

THRESHOLDS = np.linspace(0.1, 0.9, 100)

if __name__ == "__main__":

    model = Unet(
        backbone_name=BACKBONE,
        input_shape=(None, None, 3),
        classes=1,
        #         activation=activation,
        encoder_weights=None,  # 'imagenet',
        weights="models/2020-12-28_19:57:05_efficientnetb3/model.23-0.51.h5"
        # MODEL_PATH,
    )

    image_list = sorted(os.listdir("data/raw/val_masks"))

    real_masks = []
    pred_masks = []
    for image in image_list:
        rm = cv2.imread(
            os.path.join("data/raw/val_masks", image), cv2.COLOR_RGB2GRAY
        )[..., ::-1]
        pm = cv2.imread(
            os.path.join("data/raw/val", image), cv2.COLOR_RGB2GRAY
        )[..., ::-1]

        pm = np.expand_dims(pm, axis=0)
        pm = model(pm)

        real_masks.append(rm.mean(axis=-1) / 255)
        pred_masks.append(pm.numpy().squeeze(axis=-1).squeeze(axis=0))

    real_masks = np.array(real_masks, dtype=np.float32)
    pred_masks = np.array(pred_masks, dtype=np.float32)

    print(real_masks.shape, pred_masks.shape)

    np.save("real_masks", real_masks)
    np.save("pred_masks", pred_masks)

    real_masks = np.load("real_masks.npy")
    pred_masks = np.load("pred_masks.npy")

    scores = []
    for threshold in THRESHOLDS:

        pred_masks_thresholded = pred_masks.copy()

        pred_masks_thresholded[pred_masks_thresholded < threshold] = 0
        pred_masks_thresholded[pred_masks_thresholded >= threshold] = 1

        scores.append(DiceLoss()(real_masks, pred_masks_thresholded))

    with open("scores.txt", "w") as f:
        for t, s in zip(THRESHOLDS, scores):
            f.write(f"threshold: {t}, dice: {s}\n")

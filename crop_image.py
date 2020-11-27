import os

import cv2

from predict import Predictor

IMAGE_FOLDER = os.path.join("data", "raw", "test")
MASK_FOLDER = os.path.join("data", "raw", "test_masks")
OUTPUT_IMAGE_FOLDER = os.path.join("data", "raw", "test_crops")
OUTPUT_MASK_FOLDER = os.path.join("data", "raw", "test_crops_masks")

if __name__ == "__main__":

    images = os.listdir(IMAGE_FOLDER)
    masks = os.listdir(MASK_FOLDER)

    for image, mask in zip(images, masks):

        # cutting image
        large_image = cv2.imread(os.path.join(IMAGE_FOLDER, image))
        preds = Predictor.crop(large_image, size=(256, 256))

        for num, pred in enumerate(preds):
            Predictor.save_image(
                pred,
                os.path.join(
                    OUTPUT_IMAGE_FOLDER, f'{image.split(".")[0]}_{num}.jpg'
                ),
            )

        del preds
        del large_image

        # cutting mask
        large_image = cv2.imread(os.path.join(MASK_FOLDER, mask))
        preds = Predictor.crop(large_image, size=(256, 256))

        for num, pred in enumerate(preds):
            Predictor.save_image(
                pred,
                os.path.join(
                    OUTPUT_MASK_FOLDER, f'{mask.split(".")[0]}_{num}.jpg'
                ),
            )

        del preds
        del large_image

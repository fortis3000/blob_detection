import os
import typing
import pickle

import cv2
import numpy as np

from dataset import visualize, denormalize, Dataset
from config import (
    BACKBONE,
    activation,
    SAMPLE,
    MODEL_PATH,
    IMAGE_FOLDER,
    MASK_FOLDER,
    OUTPUT_MASK_FOLDER,
)
from blob_detector import BlobDetector, SkimageBlobDetector, OpenCVBlobDetector


class Predictor:
    def __init__(self, model, if_crop: bool, width: int, height: int):
        self.model = model
        self.if_crop = if_crop
        self.width = width
        self.height = height

    @classmethod
    def crop(cls, image: np.ndarray, size: tuple):
        """
        Cuts image into pieces with the size provided from left to right,
        from up to bottom.
        Pads the right pictures if needed.
        :param image: image object to cut
        :param size: tuple (h, w) of the size desired
        :return: list if np.array pieces
        """

        assert image.shape[0] > size[0], f"Image is too small {image.shape}"
        assert image.shape[1] > size[1], f"Image is too small {image.shape}"

        height, width = image.shape[:2]

        pad_horizontal = width % size[0] > 0
        pad_vertical = height % size[1] > 0

        count_horizontal = width // size[0] + int(pad_horizontal)
        count_vertical = height // size[1] + int(pad_vertical)

        pieces = []

        for i in range(count_vertical):  # iterate over y axis
            for j in range(count_horizontal):  # iterate over x axis
                current_piece = image[
                    i * size[0] : (i + 1) * size[0],
                    j * size[1] : (j + 1) * size[1],
                    :,
                ]

                if current_piece.shape[0] != size[0]:  # height
                    if current_piece.shape[1] != size[1]:  # height and width
                        current_piece = cv2.copyMakeBorder(
                            src=current_piece,
                            top=0,
                            bottom=size[0] - current_piece.shape[0],
                            left=0,
                            right=size[1] - current_piece.shape[1],
                            borderType=0,
                        )
                    else:  # only height
                        current_piece = cv2.copyMakeBorder(
                            src=current_piece,
                            top=0,
                            bottom=size[0] - current_piece.shape[0],
                            left=0,
                            right=0,
                            borderType=0,
                        )
                else:  # no height
                    if current_piece.shape[1] != size[1]:  # width only
                        current_piece = cv2.copyMakeBorder(
                            src=current_piece,
                            top=0,
                            bottom=0,
                            left=0,
                            right=size[1] - current_piece.shape[1],
                            borderType=0,
                        )
                    else:  # no padding
                        pass

                pieces.append(current_piece)
        return pieces

    @classmethod
    def detect_blobs(
        cls,
        image: np.array,
        detector: BlobDetector,
        ext_params: typing.Dict = None,
    ):
        """

        :param image:
        :return:
        """

        keypoints = detector.detect(image, ext_params=ext_params)
        return keypoints

    @classmethod
    def save_image(cls, image: np.array, filename: str):
        if image.max() > 1:
            cv2.imwrite(filename, image)
        else:
            cv2.imwrite(filename, 255 * image)

    def predict(self, image: np.array):
        """
        Predict white masks of objects detected
        :param image: np array
            Image to predict
        :return: np.array
            Predicted mask
        """
        return self.model.predict(
            image.reshape(1, self.height, self.width, -1)
        ).squeeze(axis=0)

    def predict_dataset(
        self,
        dataset: Dataset,
        output_folder: str = None,
        if_blobs: bool = False,
        review: bool = False,
        binary_threshold: float = None,
    ):
        elements_count = len(dataset)
        for i in range(elements_count):
            image, mask_true = dataset[i]

            # mask_pred is an np array of shape (256, 256, 1)
            # with values in range(0,1)
            mask_pred = (self.predict(image) * 255).astype(np.uint8)

            # binary mask if needed
            if binary_threshold:
                mask_pred = ((mask_pred > binary_threshold) * 255).astype(
                    np.uint8
                )

            # find blobs on prediction
            if if_blobs:
                detector = SkimageBlobDetector(images=None)
                # detector = OpenCVBlobDetector(images=None)
                try:
                    with open(
                        f"best_blob_params_{detector.name}.pickle", "rb"
                    ) as f:
                        ext_params = pickle.load(f)

                except Exception as e:
                    print(e)
                    ext_params = {}

                try:
                    noise_threshold = ext_params.pop("noise_threshold")
                    # filter little dark gray noise
                    mask_pred[mask_pred < noise_threshold] = 0

                    min_radius = ext_params.pop("min_radius")
                    max_radius = ext_params.pop("max_radius")
                except Exception as e:
                    print("Params not found", e)
                    min_radius = 0
                    max_radius = np.inf

                # invert colors only for detector
                keypoints = self.detect_blobs(
                    image=255 - mask_pred
                    if detector.name == "opencv"
                    else mask_pred,
                    detector=detector,
                    ext_params=ext_params,
                )

                keypoints_filtered = detector.filter_keypoints(
                    keypoints=keypoints,
                    min_radius=min_radius,
                    max_radius=max_radius,
                )

                mask_pred = detector.draw_blobs(
                    image=mask_pred,
                    keypoints=keypoints_filtered,
                )

            # obtaining filename for saving if needed
            (base_name, file_format,) = os.path.split(dataset.masks_fps[i])[
                -1
            ].split(".")

            save_name = os.path.join(
                output_folder if output_folder else "",
                f"{base_name}"
                f"{'_' + str(i) if self.if_crop else ''}"
                f"{'_' + str(len(keypoints_filtered)) if if_blobs else ''}"
                f"{'_blobs' if if_blobs else ''}."
                f"{file_format}",
            )

            print(save_name)

            # mode for saving image
            if review:
                visualize(
                    save_name=save_name,
                    image=denormalize(image.squeeze()),
                    mask_true=mask_true,
                    mask_pred=mask_pred,
                )
            else:
                self.save_image(
                    mask_pred,
                    filename=save_name,
                )


if __name__ == "__main__":
    from segmentation_models import (
        Unet,
    )

    model = Unet(
        backbone_name=BACKBONE,
        input_shape=(None, None, 3),
        classes=1,
        activation=activation,
        encoder_weights=None,  # 'imagenet',
        weights="models/2021-01-14_20:35:00_efficientnetb4/model.16-0.54.h5",  # MODEL_PATH,
    )

    predictor = Predictor(
        model=model,
        if_crop=SAMPLE == "test",
        width=256,
        height=256,
    )

    test_dataset = Dataset(
        images_dir=IMAGE_FOLDER,
        masks_dir=MASK_FOLDER,
        classes=["bulk"],
    )

    if SAMPLE == "test":
        with open("best_blob_params_skimage.pickle", "rb") as f:
            binary_threshold = pickle.load(f)["binary_threshold"]
    else:
        binary_threshold = None

    predictor.predict_dataset(
        dataset=test_dataset,
        output_folder=OUTPUT_MASK_FOLDER,
        if_blobs=SAMPLE == "test_crops",
        review=SAMPLE == "test_crops",
        binary_threshold=binary_threshold,
    )

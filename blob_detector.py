"""
Tunes OpenCV BlobDetector parameters for the best performance
Contains learnable blob detectors
"""
import os
import typing
import pickle
import numpy as np

import cv2
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from hyperopt import hp, tpe, fmin

from config import OUTPUT_MASK_FOLDER, TRAIN_ANNOTATION_PATH
from masks import ReaderAnnotation


class BlobDetector:
    def __init__(self, images):
        self.images = images
        self.SPACE = {}
        self.params = {}

    def _MSE(self, y_pred, y_true):
        """Mean squared error function"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def _MAPE(self, y_pred, y_true):
        """Simplified MAPE"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        with_nans = (y_true - y_pred) / y_true
        with_nans[with_nans != with_nans] = 0
        return np.mean(np.abs(with_nans)) * 100

    def _SMAPE(self, y_pred, y_true):
        """Simplified SMAPE"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        with_nans = (y_true - y_pred) / ((y_true + y_pred) / 2)
        with_nans[with_nans != with_nans] = 0
        return np.mean(np.abs(with_nans)) * 100

    def _build_whole_configs(self, changed_params):
        """Rewrite to update self.params method"""
        blob_params_current = self.params.copy()
        blob_params_current.update(changed_params)
        blob_params_current = {
            key: (int(val) if not isinstance(val, bool) else val)
            for key, val in blob_params_current.items()
        }
        return blob_params_current

    def detect(self, image: np.array, *args, **kwargs):
        pass

    def count_blobs(self, keypoints):
        return len(keypoints)

    def draw_blobs(self, image: np.array, keypoints, *args, **kwargs):
        pass


class OpenCVBlobDetector(BlobDetector):
    def __init__(self, images: typing.List = None):
        super(OpenCVBlobDetector, self).__init__(images)

        self.name = "opencv"
        self.SPACE = {
            "minThreshold": hp.choice("minThreshold", list(range(130, 180))),
            "maxThreshold": hp.choice("maxThreshold", list(range(180, 255))),
            "minDistBetweenBlobs": hp.choice(
                "minDistBetweenBlobs", list(range(10, 100))
            ),
            "minArea": hp.choice(
                "minArea", list(range(256 * 256 // (256 * 4), 256 * 256 // 64))
            ),
            "maxArea": hp.choice(
                "maxArea", list(range(256 * 256 // 32, 256 * 256 // 8))
            ),
            "binary_threshold": hp.choice(
                "binary_threshold", list(range(128, 240))
            ),
        }
        self.params = {
            "thresholdStep": 10,
            "minThreshold": 190,
            "maxThreshold": 256,
            "minRepeatability": 1,  # to find all blobs,
            "minDistBetweenBlobs": 0,  # pixels,
            "filterByColor": False,  # BROKEN!!!
            "blobColor": 255,
            "filterByArea": True,
            "minArea": 256 * 256 // (256 * 2),
            "maxArea": 256 * 256 // 8,
            "filterByCircularity": False,  # from 0 to 1
            "minCircularity": 0,
            "maxCircularity": 1,
            "filterByInertia": True,  # from 0 to 1
            "minInertiaRatio": 0,
            "maxInertiaRatio": 1e37,
            "filterByConvexity": True,
            "minConvexity": 0.95,
            "maxConvexity": 1e37,
        }

    def init_blob_detector(self, ext_params: dict = None):
        params = cv2.SimpleBlobDetector_Params()
        if ext_params:
            # Disable unwanted filter criteria params to detect on binary image
            params.thresholdStep = ext_params["thresholdStep"]
            params.minThreshold = ext_params["minThreshold"]
            params.maxThreshold = ext_params["maxThreshold"]
            params.minRepeatability = ext_params["minRepeatability"]
            params.minDistBetweenBlobs = ext_params["minDistBetweenBlobs"]
            params.filterByColor = ext_params["filterByColor"]
            params.blobColor = ext_params["blobColor"]
            params.filterByArea = ext_params["filterByArea"]
            params.minArea = ext_params["minArea"]
            params.maxArea = ext_params["maxArea"]
            params.filterByCircularity = ext_params["filterByCircularity"]
            params.minCircularity = ext_params["minCircularity"]
            params.maxCircularity = ext_params["maxCircularity"]
            params.filterByInertia = ext_params["filterByInertia"]
            params.minInertiaRatio = ext_params["minInertiaRatio"]
            params.maxInertiaRatio = ext_params["maxInertiaRatio"]
            params.filterByConvexity = ext_params["filterByConvexity"]
            params.minConvexity = ext_params["minConvexity"]
            params.maxConvexity = ext_params["maxConvexity"]

        ver = (cv2.__version__).split(".")
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        return detector

    def objective(self, params):

        if not self.images:
            raise ValueError("Images not provided")

        binary_threshold = params.pop("binary_threshold")
        detector = self.init_blob_detector(
            self._build_whole_configs(changed_params=params)
        )

        counts = []
        for key in self.images:
            print(key)
            image_array = cv2.imread(
                os.path.join(OUTPUT_MASK_FOLDER, key), cv2.COLOR_RGB2GRAY
            )[..., ::-1]

            # binarize mask
            binary_mask = ((image_array > binary_threshold) * 255).astype(
                np.uint8
            )

            # invert color only for detector
            keypoints = detector.detect(255 - binary_mask)
            counts.append(len(keypoints))

        print(
            f"Mean counts: {np.mean(counts):.2f},"
            f" Mean ground truth: {np.mean(true_labels):.2f}"
        )
        return self._MAPE(counts, true_labels)

    def detect(self, image: np.array, ext_params: dict = None, **kwargs):
        detector = self.init_blob_detector(ext_params=ext_params)
        keypoints = detector.detect(image)
        return keypoints

    def draw_blobs(self, image: np.array, keypoints: typing.List, **kwargs):
        """
        Causes Segmentation fault
        Blobs are generally assumed to be gray/black
        http://amroamroamro.github.io/mexopencv/opencv/detect_blob_demo.html
        https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        :param image:
        :param keypoints:
        :param kwargs:
        :return:
        """
        return cv2.drawKeypoints(
            image,
            keypoints,
            np.array([]),
            (100, 0, 100),  # purple
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )


class SkimageBlobDetector(BlobDetector):
    def __init__(self, images: list = None):
        super(SkimageBlobDetector, self).__init__(images)

        self.name = "skimage"
        self.SPACE = {
            "max_sigma": hp.choice("max_sigma", list(range(30, 50))),
            "min_sigma": hp.choice("min_sigma", list(range(5, 30))),
            "threshold": hp.uniform("threshold", 0.05, 0.5),
            "overlap": hp.uniform("overlap", 0.05, 0.5),
            "binary_threshold": hp.choice(
                "binary_threshold", list(range(128, 240))
            ),
        }
        self.params = {
            "min_sigma": 4,
            "max_sigma": 30,
            "num_sigma": 10,
            "threshold": 0.3,
            "overlap": 0.2,
        }

    def review_methods(self, image_path):
        image = cv2.imread(image_path, cv2.COLOR_RGB2GRAY)[..., ::-1]
        image_gray = rgb2gray(image)

        blobs_log = blob_log(
            image_gray, max_sigma=50, num_sigma=10, threshold=0.1
        )

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

        blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=0.1)
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

        blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=0.01)

        blobs_list = [blobs_log, blobs_dog, blobs_doh]
        colors = ["yellow", "lime", "red"]
        titles = [
            "Laplacian of Gaussian",
            "Difference of Gaussian",
            "Determinant of Hessian",
        ]
        sequence = zip(blobs_list, colors, titles)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        for idx, (blobs, color, title) in enumerate(sequence):
            ax[idx].set_title(title)
            ax[idx].imshow(image)
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax[idx].add_patch(c)
            ax[idx].set_axis_off()

        plt.tight_layout()
        plt.show()

    def objective(self, params):

        if not self.images:
            raise ValueError("Images not provided")

        counts = []
        for key in self.images:
            print(key)
            image_array = cv2.imread(
                os.path.join(OUTPUT_MASK_FOLDER, key), cv2.COLOR_RGB2GRAY
            )[..., ::-1]

            image_array = (
                (image_array > params["binary_threshold"]) * 255
            ).astype(np.uint8)

            keypoints = blob_log(
                image_array,
                max_sigma=params["max_sigma"],
                min_sigma=params["min_sigma"],
                threshold=params["threshold"],
                overlap=params["overlap"],
                num_sigma=2,
            )
            counts.append(self.count_blobs(keypoints))

        print(
            f"Mean counts: {np.mean(counts):.2f},"
            f" Mean ground truth: {np.mean(true_labels):.2f}"
        )
        return self._MSE(counts, true_labels)

    def detect(self, image: np.array, ext_params: typing.Dict = None, **kwargs):
        """

        :param **kwargs:
        :type ext_params: typing.Dict
            Updates params of the detector
        """
        updated_params = self._build_whole_configs(changed_params=ext_params)
        keypoints = blob_log(image, **updated_params)
        return keypoints

    def draw_blobs(self, image: np.array, keypoints: np.array, **kwargs):
        # make radius
        # keypoints[:, 2] = keypoints[:, 2] * np.sqrt(2)

        for blob in keypoints:
            # print(blob)
            if blob is not None:
                y, x, _, r = blob
                cv2.circle(
                    img=image,
                    center=(int(x), int(y)),
                    radius=int(r),
                    color=(100, 0, 100),
                )

        return image


if __name__ == "__main__":

    # Image loading and preparing
    train_reader = ReaderAnnotation(TRAIN_ANNOTATION_PATH)
    annotated_images = train_reader.annotation.keys()

    all_images = {}
    for key in annotated_images:

        temp = {}
        file_extension = ".JPG"
        if not os.path.isfile(
            os.path.join(OUTPUT_MASK_FOLDER, key.split(".")[0] + file_extension)
        ):
            file_extension = file_extension.lower()

        image_name = os.path.join(
            OUTPUT_MASK_FOLDER, key.split(".")[0] + file_extension
        )

        # image_name: count
        all_images[os.path.split(image_name)[-1]] = len(
            train_reader.get(key)["regions"]
        )

    images_needed = sorted(os.listdir(OUTPUT_MASK_FOLDER))

    true_labels = [all_images[filename] for filename in images_needed]

    # Detector learning
    detector = SkimageBlobDetector(images=images_needed)

    best = fmin(
        fn=detector.objective,
        space=detector.SPACE,
        algo=tpe.suggest,
        max_evals=20,
    )

    with open(f"best_blob_params_{detector.name}.pickle", "wb") as f:
        pickle.dump(
            detector._build_whole_configs(changed_params=best),
            f,
        )

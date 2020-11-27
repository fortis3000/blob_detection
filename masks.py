import json
import os
import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import (
    TRAIN_IMAGE_FOLDER,
    TRAIN_MASK_FOLDER,
    TRAIN_ANNOTATION_PATH,
    TEST_MASK_FOLDER,
    TEST_ANNOTATION_PATH,
    TEST_IMAGE_FOLDER,
)


class MaskCreator:
    def __init__(self):
        pass

    @classmethod
    def create_circular_mask(
        cls,
        h: int,
        w: int,
        center: typing.Tuple = None,
        radius: int = None,
    ) -> np.array:
        """
        Creates circular mask from params provided
        https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

        :param h: height of the image
        :param w: width of the image
        :param center: tuple of coordinates (x,y) for center point of mask
        :param radius: radial distance from center
        :return: np.array
        """

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
            # use the smallest distance between the center and image walls
            # if (radius is None):
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    @classmethod
    def _create_empty_mask(cls, image: np.array, filename: str = None):
        shape = image.shape[:-1]
        mask = np.zeros(shape=shape, dtype=np.uint8)
        if filename:
            cv2.imwrite(filename, mask)
        return mask

    @classmethod
    def _unite_masks(cls, masks: typing.List) -> np.ndarray:
        return np.clip(sum(masks), a_min=0, a_max=1)

    @classmethod
    def visualize(
        cls,
        image: np.array,
        masks: typing.List,
        filename: str = None,
        use_image: bool = False,
    ) -> np.ndarray:
        """
        Applies masks to the picture
        :param image: np.array
        :param masks: list of np.arrays
        :param filename: str
        :param use_image: if use initial image to create masked image
        :return: np.array
        """

        common_mask = cls._unite_masks(masks)

        if use_image:
            common_mask = np.array(
                image * common_mask[:, :, np.newaxis], dtype=np.uint8
            )

        assert len(np.unique(common_mask)) < 3

        if filename:
            # *255 to correct grayscale
            cv2.imwrite(filename, common_mask * int(255))

        plt.imshow(common_mask)
        plt.close()


class ReaderAnnotation:
    def __init__(self, path: str = ""):
        with open(path, "r") as f:
            self.annotation = json.load(fp=f)

    def get(self, image_id: str) -> typing.Dict:
        """
        Queries annotation from file for image provided
        :param image_id: image key in annotation file
        :return: dict {
            'fileref': '',
            'size': 1651,
            'filename': 'Photo (315)_15.JPG',
            'base64_img_data': '',
            'file_attributes': {},
            'regions': {}
            }
        """
        return self.annotation.get(image_id)

    def get_radius_min(self, regions: typing.Dict, base_radius=10) -> int:

        if not regions:
            return -1
        elif len(regions) == 1:
            return base_radius

        centers = []
        for (key, val) in regions.items():
            centers.append(
                [
                    val["shape_attributes"]["cx"],
                    val["shape_attributes"]["cy"],
                ]
            )

        x = np.array(centers)
        x_sq = np.sum(x ** 2, axis=1)
        matmul = np.matmul(x, x.T)

        dists = np.sqrt(x_sq - 2 * matmul + x_sq[:, np.newaxis])  # broadcasting
        dists[dists == 0] = np.inf

        return int(np.min(dists))


def create_masks(image_folder: str, annotation_path: str, outpath: str):
    """
    Places masks in folders
    :param image_folder: path to the images
    :param annotation_path: path to the masks
    :param outpath: path to output directory
    :return:
    """

    train_reader = ReaderAnnotation(annotation_path)

    all_images = os.listdir(image_folder)
    annotated_images = train_reader.annotation.keys()

    creator = MaskCreator()

    for key in annotated_images:
        file_extension = ".JPG"
        if not os.path.isfile(
            os.path.join(
                image_folder,
                key.split(".")[0] + file_extension,
            )
        ):
            file_extension = file_extension.lower()

        image_name = os.path.join(
            image_folder,
            key.split(".")[0] + file_extension,
        )
        print(image_name)

        out_image_path = os.path.join(outpath, os.path.split(image_name)[-1])
        assert os.path.exists(out_image_path), "Out image path doesn't exist"

        image = plt.imread(image_name)
        h, w, c = image.shape

        regions = train_reader.get(key)["regions"]
        # less than minimal distance
        radius = int(train_reader.get_radius_min(regions=regions) * 0.9)

        masks = []
        for _, center in regions.items():
            masks.append(
                creator.create_circular_mask(
                    h=h,
                    w=w,
                    center=(
                        int(center["shape_attributes"]["cx"]),
                        int(center["shape_attributes"]["cy"]),
                    ),
                    radius=radius,
                )
            )

            if len(masks) > 50:
                masks = [creator._unite_masks(masks)]

        if masks:
            creator.visualize(
                image=image,
                masks=masks,
                filename=out_image_path,
                use_image=False,
            )
        else:
            creator._create_empty_mask(image=image, filename=out_image_path)

    print("Empty images:")
    for empty_image in list(set(all_images) - set(annotated_images)):
        if os.path.exists(out_image_path):
            continue
        empty_image = os.path.join(image_folder, empty_image)
        print(empty_image)
        image = plt.imread(empty_image)
        creator._create_empty_mask(
            image=image,
            filename=os.path.join(
                outpath,
                os.path.split(empty_image)[-1],
            ),
        )


def main():
    create_masks(TRAIN_IMAGE_FOLDER, TRAIN_ANNOTATION_PATH, TRAIN_MASK_FOLDER)
    create_masks(TEST_IMAGE_FOLDER, TEST_ANNOTATION_PATH, TEST_MASK_FOLDER)


if __name__ == "__main__":
    main()

import os
from math import factorial
import albumentations as A


class Augmentor:
    def __init__(self):
        pass

    @staticmethod
    def round_clip_0_1(x, **kwargs):
        return x.round().clip(0, 1)

    def _get_base_probability(self, n: int) -> int:
        """
        Uniforms events of all augmentation occurred
        :param n:
        :return:
        """

        counts = 0
        for i in range(n):
            counts += factorial(n) // factorial(i) // factorial(n - i)

        return counts

    def get_training_augmentation(self):
        augs_count = 11
        base_proba = self._get_base_probability(n=augs_count)

        train_transform = [
            # simple
            A.HorizontalFlip(p=base_proba),
            A.VerticalFlip(p=base_proba),
            A.Rotate(limit=(-180, 180), p=base_proba),
            A.Transpose(p=base_proba),
            # cutting and scaling
            A.OneOf(
                [
                    A.RandomResizedCrop(
                        height=256,
                        width=256,
                        scale=(0.4, 1.0),
                        ratio=(0.8, 2.0),
                        p=base_proba,
                    ),
                    A.IAAPerspective(
                        scale=(0.1, 0.2),
                        keep_size=True,
                        always_apply=False,
                        p=base_proba,
                    ),
                ],
                p=base_proba,
            ),
            # colour
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.6),
                contrast_limit=(-0.3, 0.6),
                brightness_by_max=True,
                always_apply=False,
                p=base_proba,
            ),
            A.HueSaturationValue(
                hue_shift_limit=(-360, 360),
                sat_shift_limit=(-80, 80),
                val_shift_limit=(-80, 80),
                p=base_proba,
            ),
            A.ToGray(p=base_proba),
            A.RGBShift(
                r_shift_limit=(100, 200),
                g_shift_limit=(100, 200),
                b_shift_limit=(100, 200),
                always_apply=False,
                p=base_proba,
            ),
            # noise
            A.OneOf(
                [
                    A.GridDistortion(
                        num_steps=16,
                        distort_limit=(-0.5, 0.5),
                        p=base_proba,
                    ),
                    A.GaussianBlur(
                        blur_limit=(5, 11),
                        sigma_limit=0,
                        p=base_proba,
                    ),
                    A.GaussNoise(
                        var_limit=(50, 75),
                        mean=0,
                        p=base_proba,
                    ),
                    A.Blur(
                        blur_limit=(7, 13),
                        p=base_proba,
                    ),
                ],
                p=base_proba,
            ),
            A.MotionBlur(
                blur_limit=10,
                p=base_proba,
            ),
            A.Lambda(mask=self.round_clip_0_1),
            A.Resize(
                height=256,
                width=256,
                p=1,
            ),
        ]

        return A.Compose(
            train_transform,
            p=base_proba,
        )

    def get_validation_augmentation(self):
        test_transform = []
        return A.Compose(test_transform)

    def get_preprocessing(self, preprocessing_fn):
        _transform = [A.Lambda(image=preprocessing_fn)]
        return A.Compose(_transform)


if __name__ == "__main__":
    from dataset import Dataset, visualize

    data_path = "data"
    dataset = Dataset(
        images_dir=os.path.join(data_path, "test"),
        masks_dir=os.path.join(data_path, "train_annotations"),
        classes=["bulk"],
        augmentations=Augmentor.get_training_augmentation(),
        preprocessing=None,
    )

    image, mask = dataset[5]

    visualize(image=image, bulk_mask=mask[..., 0].squeeze())

"""
Bulk dataloader module
"""
import numpy as np
from tensorflow import (
    keras,
)

from dataset import Dataset


class Dataloader(keras.utils.Sequence):
    def __init__(
        self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, item: int):
        """
        Coollects batch data
        :param item:
        :return:
        """
        start = item * self.batch_size
        stop = (item + 1) * self.batch_size

        data = []
        for i in range(start, stop):
            data.append(self.dataset[i])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indices each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


if __name__ == "__main__":
    import os

    data_path = os.path.join("data", "raw")
    ds = Dataset(
        images_dir=os.path.join(data_path, "train"),
        masks_dir=os.path.join(data_path, "train_masks"),
        classes=["bulk"],
        augmentations=None,
        preprocessing=None,
    )

    loader = Dataloader(ds, batch_size=2, shuffle=False)

    print(loader[1])
    print(loader[1][0].shape)
    print(loader[1][1].shape)
    print(loader[1][1].max())

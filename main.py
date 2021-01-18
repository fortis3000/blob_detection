import os
import datetime as dt

import matplotlib.pyplot as plt
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import DiceLoss, BinaryFocalLoss
from segmentation_models.metrics import iou_score
from tensorflow import keras

from augs import Augmentor
from config import (
    TRAIN_IMAGE_FOLDER,
    TRAIN_MASK_FOLDER,
    VAL_IMAGE_FOLDER,
    VAL_MASK_FOLDER,
    BACKBONE,
    MODELS_FOLDER,
    activation,
    BATCH_SIZE,
    CLASSES,
    LR,
    EPOCHS,
)
from dataloader import Dataloader
from dataset import Dataset

augmentor = Augmentor()


def get_dt_str():
    """Returns current date and time in string format"""
    return str(dt.datetime.now()).split(".")[0].replace(" ", "_")


def plot_history(history):
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history["iou_score"])
    plt.plot(history.history["val_iou_score"])
    plt.title("Model iou_score")
    plt.ylabel("iou_score")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.savefig("history.jpg")
    plt.close()


DATESTRING = get_dt_str()

CUR_MODEL_FOLDER = os.path.join(MODELS_FOLDER, DATESTRING + "_" + BACKBONE)

os.mkdir(CUR_MODEL_FOLDER)

preprocess_input = get_preprocessing(BACKBONE)

model = Unet(
    backbone_name=BACKBONE,
    input_shape=(None, None, 3),
    classes=1,
    activation=activation,
    encoder_weights=None,  # 'imagenet', # None
    # encoder_features=[0, 1, 2],
    # encoder_freeze=False
)
optim = keras.optimizers.Adam(LR)
dice_loss = DiceLoss()
total_loss = DiceLoss() + BinaryFocalLoss()
model.compile(
    optim,
    loss=total_loss,
    metrics=[iou_score],
)

# data flow initialization
train_dataset = Dataset(
    images_dir=TRAIN_IMAGE_FOLDER,
    masks_dir=TRAIN_MASK_FOLDER,
    classes=CLASSES,
    augmentations=augmentor.get_training_augmentation(),
    preprocessing=None,  # get_preprocessing(preprocess_input)
)

val_dataset = Dataset(
    images_dir=VAL_IMAGE_FOLDER,
    masks_dir=VAL_MASK_FOLDER,
    classes=CLASSES,
    augmentations=None,  # get_validation_augmentation(),
    preprocessing=None,  # get_preprocessing(preprocess_input) # use for TTA
)

train_dataloader = Dataloader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_dataloader = Dataloader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# check shapes for errors
print(train_dataloader[0][0].shape)
print(train_dataloader[0][1].shape)

# assert train_dataloader[0][0].shape == (BATCH_SIZE, 256, 256, 3)
# assert train_dataloader[0][1].shape == (BATCH_SIZE, 256, 256, n_classes)
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=15,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            CUR_MODEL_FOLDER, "model.{epoch:02d}-{val_loss:.2f}.h5"
        ),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
        options=None,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=10,
        min_delta=0.0001,
        cooldown=0,
        min_lr=0.000001,
        verbose=1,
        mode="min",
    ),
    keras.callbacks.TensorBoard(log_dir=os.path.join("logs", DATESTRING)),
]


if __name__ == "__main__":

    # fit model
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=val_dataloader,
        validation_steps=len(val_dataloader),
    )

    # Saving trained model
    model.save(CUR_MODEL_FOLDER)
import os

BACKBONE = "efficientnetb3"  # backbones_factory.py

# data params
DATAPATH = os.path.join("data", "raw")
TRAIN_IMAGE_FOLDER = os.path.join(DATAPATH, "train")
VAL_IMAGE_FOLDER = os.path.join(DATAPATH, "val")
TEST_IMAGE_FOLDER = os.path.join(DATAPATH, "test")
TRAIN_MASK_FOLDER = os.path.join(DATAPATH, "train_masks")
VAL_MASK_FOLDER = os.path.join(DATAPATH, "val_masks")
TEST_MASK_FOLDER = os.path.join(DATAPATH, "test_masks")
TRAIN_ANNOTATION_PATH = os.path.join(
    DATAPATH, "train_annotations", "train_labels.json"
)
TEST_ANNOTATION_PATH = os.path.join(
    DATAPATH, "test_annotations", "test_labels.json"
)

# Prediction
# TODO: replace with CLI params
SAMPLE = "val"
MODEL_PATH = os.path.join("models", BACKBONE, "h5")
IMAGE_FOLDER = os.path.join(DATAPATH, f"{SAMPLE}")
MASK_FOLDER = os.path.join(DATAPATH, f"{SAMPLE}_masks")
OUTPUT_MASK_FOLDER = os.path.join(DATAPATH, f"{SAMPLE}_masks_pred")

# model params
CLASSES = ["bulk"]
# binary and multiclass segmentation
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
# define model
activation = "relu" if n_classes == 1 else "softmax"

# train params
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 200

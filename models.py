from tensorflow import keras
from tensorflow.keras import layers
import ml_collections


def get_shallow_cnn(config: ml_collections.ConfigDict) -> keras.Model:
    """
    Creates a shallow CNN model for the CIFAR-10 dataset.

    Reference: https://keras.io/examples/vision/semisupervised_simclr/
    """
    dataset_config = config["dataset_config"]
    image_size = dataset_config["image_size"]
    width = config["model_width"]

    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, 3)),
            layers.Rescaling(scale=1.0 / 255),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            layers.GlobalAvgPool2D(),
            layers.Dense(width, activation="relu"),
            layers.Dense(dataset_config["num_classes"], activation="softmax"),
        ],
        name=f"{dataset_config['dataset_name']}_classifier",
    )

import ml_collections


def get_dataset_config(
    dataset_name: str, num_classes: int, image_size: int
) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.dataset_name = dataset_name
    config.num_classes = num_classes
    config.image_size = image_size
    return config.lock()


def get_model_config(
    dataset_name: str, num_classes: int, image_size: int, path: str
) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.dataset_config = get_dataset_config(dataset_name, num_classes, image_size)

    config.model_width = 128
    config.training_epochs = 10
    config.batch_size = 512
    config.path = path
    return config.lock()

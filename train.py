import configs
import models
import logging
import tensorflow as tf

from datetime import datetime


AUTO = tf.data.AUTOTUNE


def main():
    # Fetch the run configs.
    timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")
    run_config = configs.get_model_config(
        dataset_name="cifar10", num_classes=10, image_size=32, path=timestamp
    )
    dataset_config = run_config["dataset_config"]
    with run_config.unlocked():
        run_config.path = f"{timestamp}-{dataset_config.dataset_name}"

    # Download the dataset.
    logging.info("Loading dataset.")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_val, y_val) = (
        (x_train[:40000], y_train[:40000]),
        (x_train[40000:], y_train[40000:]),
    )
    logging.info(f"Training samples: {len(x_train)}")
    logging.info(f"Validation samples: {len(x_val)}")
    logging.info(f"Testing samples: {len(x_test)}")

    # Prepare dataloader.
    logging.info("Preparing TensorFLow dataset objects.")
    batch_size = run_config["batch_size"]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(batch_size * 100).batch(batch_size).prefetch(AUTO)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(AUTO)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(AUTO)

    # Initialize model.
    logging.info("Preparing model.")
    model = models.get_shallow_cnn(run_config)
    model.compile(
        loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )

    # Initialize callbacks.
    logging.info("Preparing callbacks.")
    checkpoint_filepath = f"{run_config.path}/checkpoints"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{run_config.path}/logs")

    # Train the model.
    logging.info("Training model.")
    _ = model.fit(
        train_ds,
        epochs=run_config.training_epochs,
        validation_data=val_ds,
        callbacks=[checkpoint_callback, tb_callback],
    )

    # Evaluate the model.
    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_ds, callbacks=[tb_callback])
    logging.info(f"Test accuracy: {round(accuracy * 100, 2)}%")


if __name__ == "__main__":
    main()

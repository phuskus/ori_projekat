import tensorflow as tf
import models.kerasTutorial
import models.resnet50
import matplotlib.pyplot as plt
import util

if __name__ == '__main__':
    # Sort raw xrays images into subfolders based on label
    # util.sortImages("dataset_raw/metadata/chest_xray_metadata.csv", "dataset_raw", "xrays")

    """
    Quick model and config selection
    """
    modelFile = models.kerasTutorial

    """
    Parameters
    """
    batch_size = modelFile.batch_size
    epochs = modelFile.epochs
    image_size = modelFile.image_size
    learning_rate = modelFile.learning_rate
    func_make_model = modelFile.makeModel
    model_data_aug = modelFile.data_augmentation
    color_mode = modelFile.color_mode
    if color_mode == "grayscale":
        num_channels = 1
    else:
        num_channels = 3

    """
    Prepare training and validation datasets
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "xrays",
        validation_split=0.2,
        subset="training",
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
        label_mode="categorical",
        color_mode=color_mode
    )


    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "xrays",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
        label_mode="categorical",
        color_mode=color_mode
    )

    # Augment the training dataset
    train_ds = train_ds.map(
        lambda x, y: (model_data_aug(x, training=True), y))

    # Buffer to prevent I/O blocking
    train_ds = train_ds.prefetch(buffer_size=batch_size)
    val_ds = val_ds.prefetch(buffer_size=batch_size)

    # Make the model
    model, callbacks = func_make_model(learning_rate=learning_rate, input_shape=image_size + (num_channels,))

    # Train the model
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds
    )

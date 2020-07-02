import pandas
import os
from shutil import copyfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def sortImages():
    csv = pandas.read_csv("dataset_raw/metadata/chest_xray_metadata.csv")
    directory = "dataset_raw"
    path = ""
    label = ""
    labels = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        try:
            label = csv.loc[csv["X_ray_image_name"] == filename].iloc[0]['Label']
        except:
            print(filename, "not found in csv")
            continue

        if label in labels.keys():
            labels[label] += 1
        else:
            labels[label] = 1

        if label == "Pnemonia":
            # Copy to Pneumonia folder
            copyfile(path, os.path.join("xrays/Pneumonia", filename))
        elif label == "Normal":
            # Copy to Normal folder
            copyfile(path, os.path.join("xrays/Normal", filename))
        else:
            # Copy to Other folder
            copyfile(path, os.path.join("xrays/Other", filename))

    print(labels)

data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

def makeModel(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Augment the image
    x = data_augmentation(inputs)

    # Enter the network
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

if __name__ == '__main__':
    # Sort raw xrays images into subfolders based on label
    # sortImages()

    batch_size = 32
    image_size = (224, 224)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "xrays",
        validation_split=0.2,
        subset="training",
        seed=1337,
        batch_size=batch_size,
        image_size=image_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "xrays",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        batch_size=batch_size,
        image_size=image_size
    )

    # Preview dataset
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(int(labels[i]))
    #         plt.axis("off")
    # plt.show()


    # Augment the training dataset
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y))

    # Buffer to prevent I/O blocking
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    # Make the model
    model = makeModel(input_shape=image_size + (3,), num_classes=2)

    # Train the model
    epochs = 50
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds
    )
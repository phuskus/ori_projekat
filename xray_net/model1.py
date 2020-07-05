import pandas
import os
from shutil import copyfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import applications

def sortImages():
    csv = pandas.read_csv("dataset_raw/metadata/chest_xray_metadata.csv")
    csv = csv.replace(np.nan, "", regex=True)
    directory = "dataset_raw"
    path = ""
    label = ""
    labels = {}
    print("Sorting images into subfolders...")
    listDir = os.listdir(directory)
    progressStep = int(0.1 * len(listDir))
    if not os.path.exists("xrays"):
        os.mkdir("xrays")

    missingCount = 0
    for idx, filename in enumerate(listDir):
        if idx % progressStep == 0:
            print(str(int(((idx+1) * 100.0) / len(listDir))) + "% complete")
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        try:
            row = csv.loc[csv["X_ray_image_name"] == filename].iloc[0]
        except:
            #print(filename, "not found in csv")
            missingCount += 1
            continue
        label = row['Label'] + row['Label_1_Virus_category']

        if label in labels.keys():
            labels[label] += 1
        else:
            labels[label] = 1

        dirPath = os.path.join("xrays", label)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        copyfile(path, os.path.join(dirPath, filename))

    print("Sorting complete!")
    print("%d filenames were not found in the CSV" % missingCount)
    print("Images copied into subdirectories:")
    print(labels)
    return len(labels)

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
    # sortImages

    """
    Parameters
    """
    batch_size = 32
    epochs = 50
    image_size = (224, 224)

    """
    Prepare training and validation datasets
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "xrays",
        validation_split=0.1,
        subset="training",
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
        label_mode="categorical",
        color_mode="grayscale"
    )


    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "xrays",
        validation_split=0.1,
        subset="validation",
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
        label_mode="categorical",
        color_mode="grayscale"
    )

    # Augment the training dataset
    #train_ds = train_ds.map(
    #    lambda x, y: (data_augmentation(x, training=True), y))

    # Buffer to prevent I/O blocking
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    # Make the model
    model = makeModel(input_shape=image_size + (1,), num_classes=3)

    # Train the model
    startLr = 1e-3

    def scheduler(epoch, lr):
        return -startLr / epochs * epoch + startLr

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=startLr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        keras.callbacks.LearningRateScheduler(scheduler)
    ]

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds
    )

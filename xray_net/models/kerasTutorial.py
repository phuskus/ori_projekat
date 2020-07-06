import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 32
epochs = 50
learning_rate = 1e-4
image_size = (224, 224)
color_mode = "grayscale"

data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

def makeModel(input_shape, learning_rate):
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
    activation = "softmax"
    units = 3 # number of classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.SGD(momentum=0.01, nesterov=True),
        loss="categorical_crossentropy",
        metrics=[
            keras.metrics.categorical_accuracy
        ]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint("./train_checkpoints/save_at_{epoch}.h5")
    ]

    return model, callbacks

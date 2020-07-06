import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint


batch_size = 8
epochs = 10
learning_rate = 1e-5
image_size = (300, 300)
color_mode = "rgb"

data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

def makeModel(input_shape, learning_rate):
    base_model = ResNet50(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)

    fc_layers = [1024, 1024]
    dropout = 0.5

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(3, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    finetune_model.compile(Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        keras.callbacks.ModelCheckpoint("train_checkpoints/save_at_{epoch}.h5")
    ]

    return finetune_model, callbacks
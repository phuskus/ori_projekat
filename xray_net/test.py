"""
Categorical Accuracy na test datasetu: 81.34%
"""

import util

#util.sortImages(csvPath="chest-xray-dataset-test/chest_xray_test_dataset.csv",
#                imgFolderPath="chest-xray-dataset-test/test",
#                outputFolderPath="xraysTest")

# load and evaluate a saved model
from tensorflow.keras.models import load_model
import tensorflow as tf

# load model
model = load_model("train_checkpoints/sgd_acc77.h5")

# summarize model.
model.summary()

# load dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "xrays",
        seed=1337,
        label_mode="categorical",
        color_mode="grayscale"
    )
# evaluate the model
score = model.evaluate(test_ds)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

#initializing the base model with required parameters
base = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        pooling=None,
        classes=1000,)

#Using Sequential method in order to add the required layers to the model
model = tf.keras.models.Sequential([
    base,
    tf.keras.layers.Dense(1, activity_regularizer=tf.keras.regularizers.L2(0.01)) #Activity_regularizer applies penalty on the OUTPUT
])
model.summary()


model.save('./model/mobileNetv2.h5')


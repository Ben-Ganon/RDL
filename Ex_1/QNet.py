import tensorflow as tf
from keras import layers


def build_model(layer_num, layer_sizes):
    for i in range(layer_num):
        if i == 0:
            model = tf.keras.Sequential([
                layers.Dense(layer_sizes[i], activation='relu')
            ])
        else:
            model.add(layers.Dense(layer_sizes[i], activation='relu'))
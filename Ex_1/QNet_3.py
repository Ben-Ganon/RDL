import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

class QNet(Model):
    def __init__(self, layer_sizes, output_size, learning_rate=0.01, optimizer='adam'):
        super().__init__()
        # initialize the layers with xavier initialization
        self.model = tf.keras.Sequential()
        self.model.add(Dense(layer_sizes[0], activation='relu', input_shape=(4,)))
        for layer in layer_sizes[1:]:
            self.model.add(Dense(layer, activation='relu'))
        self.model.add(Dense(output_size, activation='linear'))
        # Selecting the optimizer
        if optimizer == "adam":
            self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def return_model(self):
        return self.model




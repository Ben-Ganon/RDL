import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

class QNet(Model):
    def __init__(self, layer_sizes, output_size, learning_rate, optimizer):
        super().__init__()
        # initialize the layers with xavier initialization
        self.model = tf.keras.Sequential()
        self.model.add(Dense(layer_sizes[0], activation='relu', input_shape=(4,), kernel_initializer=keras.initializers.GlorotNormal()))
        for layer in layer_sizes[1:]:
            self.model.add(Dense(layer, activation='relu'))
        self.model.add(Dense(output_size, activation='linear', kernel_initializer=keras.initializers.GlorotNormal()))
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
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def return_model(self):
        return self.model

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return self.output_layer(x)




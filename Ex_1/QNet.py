import tensorflow as tf
from keras import layers
import keras
INPUT_SIZE = 4

def build_model(layer_num, layer_sizes, output_size):
    for i in range(layer_num):
        if i == 0:
            model = tf.keras.Sequential()
            model.add(layers.Dense(layer_sizes[i], activation='relu', input_shape=(INPUT_SIZE,)))
        else:
            model.add(layers.Dense(layer_sizes[i], activation='relu'))
    model.add(layers.Dense(output_size))
    return model


class QNet:
    def __init__(self, layer_num, layer_sizes, output_size, optimizer, learning_rate=0.01):
        self.model = build_model(layer_num, layer_sizes, output_size)
        if optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.model.compile(optimizer=self.optimizer, loss='mse')
        self.loss_object = keras.losses.MeanSquaredError()


    def forward(self, state):
        return self.model(state)

    def loss(self, reward, target):
        return self.loss_object(reward, target)

    def backwards(self, loss):
        variables = self.model.trainable_variables
        grads_and_vars = self.optimizer.compute_gradients(loss, variables)
        self.optimizer.apply_gradients(grads_and_vars)

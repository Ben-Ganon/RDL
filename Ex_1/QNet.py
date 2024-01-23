import tensorflow as tf
from tensorflow.keras import layers, models


class QNet(models.Model):
    def __init__(self, layer_sizes, output_size, learning_rate=0.01, optimizer='adam'):
        super(QNet, self).__init__()
        self.model_layers = [layers.Dense(size, activation='relu') for size in layer_sizes]
        self.output_layer = layers.Dense(output_size)

        # Selecting the optimizer
        if optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")

        # Compile the model
        self.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return self.output_layer(x)

    def custom_train_step(self, state, target, action):
        target = tf.reshape(target, (-1, 1))
        with tf.GradientTape() as tape:
            # get the value after call only for the action taken
            predictions = tf.gather_nd(self.call(state), action, batch_dims=1)
            loss = self.compiled_loss(target, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss





# import tensorflow as tf
# from keras import layers
# import keras
# INPUT_SIZE = 4
#
# def build_model(layer_num, layer_sizes, output_size):
#     for i in range(layer_num):
#         if i == 0:
#             model = tf.keras.Sequential()
#             model.add(layers.Dense(layer_sizes[i], activation='relu', input_shape=(INPUT_SIZE,)))
#         else:
#             model.add(layers.Dense(layer_sizes[i], activation='relu'))
#     model.add(layers.Dense(output_size))
#     return model
#
#
# class QNet(tf.Module):
#     def __init__(self, layer_num, layer_sizes, output_size, optimizer, learning_rate=0.01):
#         super(QNet, self).__init__()
#         self.model = build_model(layer_num, layer_sizes, output_size)
#         if optimizer == "adam":
#             self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#         elif optimizer == "sgd":
#             self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#         elif optimizer == "rmsprop":
#             self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
#         self.model.compile(optimizer=self.optimizer, loss='mse')
#         self.loss_object = keras.losses.MeanSquaredError()
#
#
#     def forward(self, state):
#         return self.model(state)
#
#     def loss(self, reward, target):
#         return self.loss_object(reward, target)
#
#     def backwards(self, y_i, rewards, tape):
#         loss = self.loss_object(rewards, y_i)
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#

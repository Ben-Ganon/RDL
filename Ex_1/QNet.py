import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

class QNet(Model):
    def __init__(self, layer_sizes, output_size, learning_rate=0.01, optimizer='adam'):
        super().__init__()
        # initialize the layers with xavier initialization
        self.model_layers = []
        self.model_layers.append(Dense(layer_sizes[0], activation='relu', input_shape=(4,),
                                       kernel_initializer=keras.initializers.GlorotNormal()))
        for i in range(1, len(layer_sizes)):
            self.model_layers.append(Dense(layer_sizes[i], activation='relu',
                                                      kernel_initializer=keras.initializers.GlorotNormal()))
        # output layer
        self.output_layer = Dense(output_size, kernel_initializer=keras.initializers.GlorotNormal())
        # Selecting the optimizer
        if optimizer == "adam":
            self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")
        self.loss_object = keras.losses.MeanSquaredError()
        # Compile the model
        self.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return self.output_layer(x)

def custom_train_step(model, state, target, action):
    with tf.GradientTape() as tape:
        # get the value after call only for the action taken
        output = model(state)
        predictions = tf.gather_nd(output, action, batch_dims=1)
        loss = model.loss_object(target, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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

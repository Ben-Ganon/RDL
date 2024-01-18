import gymnasium as gym
import numpy as np
from Params import Params

N = 100
LAYER_NUM = 3

Q_network = tf.keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(4)
])


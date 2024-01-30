import gymnasium as gym
import numpy as np

from Params import Params
import random
from collections import deque
from QNet_3 import QNet
import tensorflow as tf


INPUT_SIZE = 4
OUT_PUT_SIZE = 2
OPTIMIZER = "sgd"
LEARNING_RATE = 0.01
LAYER_SIZES = [16, 16, 16]
MAX_ITERATION = 3000
DISCOUNT = 0.91
UPDATE_TARGET_EVERY = 6
EPOCHS = 1000
N_REPLAY_SIZE = 500
BATCH_SIZE = 16
GREEDY_EPSILON = 0.9
GREEDY_EPSILON_DECAY = 0.5
RENDER_MODE = False
REWARD_QUE_SIZE = 100
ENV_NAME = "CartPole-v1"
MIN_GREEDY_EPSILON = 0
LOAD = False



class Q_function():
    def __init__(self, load, layers, output_size, optimizer, learning_rate):
        self.Qnet_target_A = QNet(layer_sizes=layers, output_size=output_size, optimizer=optimizer,
                                  learning_rate=learning_rate).return_model()
        self.Qnet_target_B = QNet(layer_sizes=layers, output_size=output_size, optimizer=optimizer,
                                  learning_rate=learning_rate).return_model()
        self.Qnet_values_A = QNet(layer_sizes=layers, output_size=output_size, optimizer=optimizer,
                                  learning_rate=learning_rate).return_model()
        self.Qnet_values_B = QNet(layer_sizes=layers, output_size=output_size, optimizer=optimizer,
                                  learning_rate=learning_rate).return_model()
        dummy_input = tf.keras.Input(shape=INPUT_SIZE)
        self.Qnet_target_A(dummy_input)
        self.Qnet_target_B(dummy_input)
        self.Qnet_values_A(dummy_input)
        self.Qnet_values_B(dummy_input)
        if load:
            self.LOAD_WEIGHTS(self.Qnet_target_A, self.Qnet_target_B, self.Qnet_values_A, self.Qnet_values_B)
        self.print_params()
        print(self.Qnet_target_A.summary())

    def print_params(self):
        print("Optimizer: {}, Learning Rate: {}, Layer Sizes: {}, Output Size: {}".format(OPTIMIZER, LEARNING_RATE,
                                                                                          LAYER_SIZES, OUT_PUT_SIZE))

    def LOAD_WEIGHTS(self, Qnet_target_A, Qnet_target_B, Qnet_values_A, Qnet_values_B):
        Qnet_target_A.load_weights("Weights/Qnet_target_A.weights.h5")
        Qnet_target_B.load_weights("Weights/Qnet_target_B.weights.h5")
        Qnet_values_A.load_weights("Weights/Qnet_values_A.weights.h5")
        Qnet_values_B.load_weights("Weights/Qnet_values_B.weights.h5")

    def update_target_models(self):
        dummy_input = tf.keras.Input(shape=INPUT_SIZE)
        self.Qnet_target_A(dummy_input)
        self.Qnet_target_B(dummy_input)
        self.Qnet_target_A.set_weights(self.Qnet_values_A.get_weights())
        self.Qnet_target_B.set_weights(self.Qnet_values_B.get_weights())

    def save_weights(self):
        self.Qnet_target_A.save_weights("Weights/Qnet_target_A.weights.h5")
        self.Qnet_target_B.save_weights("Weights/Qnet_target_B.weights.h5")
        self.Qnet_values_A.save_weights("Weights/Qnet_values_A.weights.h5")
        self.Qnet_values_B.save_weights("Weights/Qnet_values_B.weights.h5")


class env():
    def __init__(self, env_name, render_mode):
        if render_mode:
            self.environment = gym.make(env_name, render_mode="human")
        else:
            self.environment = gym.make(env_name, render_mode=None)
        self.observation = self.environment.observation_space
        self.action = self.environment.action_space

    def get_env(self):
        return self.environment


class replay:
    def __init__(self, n_replay_size, batch_size):
        self.que = deque(maxlen=n_replay_size)
        self.batch_size = batch_size

    def sample_batch(self):
        if len(self.que) < self.batch_size:
            return random.sample(self.que, len(self.que))
        return random.sample(self.que, self.batch_size)

    def add(self, state, action, reward, next_state, done):
        self.que.append((state, action, reward, next_state, done))

    def get_batch(self, replays):
        replay_state = np.array([replay[0] for replay in replays]).reshape(len(replays), 4)
        replay_action = np.array([replay[1] for replay in replays]).reshape(len(replays), 1)
        replay_reward = np.array([replay[2] for replay in replays]).reshape(len(replays), 1)
        replay_next_step = np.array([replay[3] for replay in replays]).reshape(len(replays), 4)
        return replay_state, replay_action, replay_reward, replay_next_step


class Params:
    def __init__(self, greedy_epsilon, greedy_epsilon_decay, max_iteration, discount, update_target_every, epochs):
        self.greedy_epsilon = greedy_epsilon
        self.greedy_epsilon_decay = greedy_epsilon_decay
        self.max_iteration = max_iteration
        self.discount = discount
        self.update_target_every = update_target_every
        self.epochs = epochs
        self.reward_que = deque(maxlen=REWARD_QUE_SIZE)
        self.accumulate_reward = 0
        self.avg_reward = 0
        self.avg_loss = 0
        self.done = False
        self.who_is_training = 0  # 1 for A, 0 for B
        self.print_params()
        pass

    def print_params(self):
        print("greedy_epsilon: {}, greedy_epsilon_decay: {}, max_iteration: {}, discount: {}, update_target_every: {}, epochs: {}".
              format(self.greedy_epsilon, self.greedy_epsilon_decay, self.max_iteration, self.discount,
                     self.update_target_every, self.epochs))
    def update_avg_reward(self):
        self.reward_que.append(self.accumulate_reward)
        self.avg_reward = np.mean(self.reward_que)

    def reset_params(self):
        self.accumulate_reward = 0
        self.avg_loss = 0
        self.done = False

    def update_greedy_epsilon(self, greedy_epsilon_decay):
        self.greedy_epsilon *= greedy_epsilon_decay
        self.greedy_epsilon = max(self.greedy_epsilon, MIN_GREEDY_EPSILON)

    def add_reward(self, reward):
        self.accumulate_reward += reward




class Agent:
    def __init__(self, ):
        # init env
        self.env = env(env_name=ENV_NAME, render_mode=RENDER_MODE).get_env()
        # init replay
        self.replay_q = replay(n_replay_size=N_REPLAY_SIZE, batch_size=BATCH_SIZE)
        # init Qnets
        self.models = Q_function(load=LOAD, layers=LAYER_SIZES, output_size=self.env.action_space.n, optimizer=OPTIMIZER,
                                 learning_rate=LEARNING_RATE)
        # init params
        self.params = Params(greedy_epsilon=GREEDY_EPSILON, greedy_epsilon_decay=GREEDY_EPSILON_DECAY, max_iteration=MAX_ITERATION, discount=DISCOUNT,
                             update_target_every=UPDATE_TARGET_EVERY, epochs=EPOCHS)

    def reset_env(self):
        state = self.env.reset()[0]
        state = np.reshape(state, (1, 4))
        self.params.reset_params()
        if self.env.render_mode:
            self.env.render()
        return state

    def sample_action(self, state):
        rand = np.random.random()
        if rand < self.params.greedy_epsilon:
            return self.env.action_space.sample()
        else:
            if rand < 0.5:
                self.params.who_is_training = 1
                return np.argmax(self.models.Qnet_target_A.predict(tf.convert_to_tensor(state), verbose=0))
            else:
                self.params.who_is_training = 0
                return np.argmax(self.models.Qnet_target_B.predict(tf.convert_to_tensor(state), verbose=0))

    def update_models(self, model_target, model_values, replays, iteration):
        y_i_array = [replay[2] if replay[4] else replay[2] + self.params.discount *
                                                 model_values.predict(tf.convert_to_tensor(replay[3]), verbose=0)[0][
                                                     replay[1]] for replay in replays]
        y_i_array = np.array(y_i_array).reshape(len(replays), 1)
        replay_state, replay_action, replay_reward, replay_next_step = self.replay_q.get_batch(replays)

        # get the value after call only for the action taken
        state = tf.convert_to_tensor(replay_state)
        with tf.GradientTape() as tape:
            output = model_target(state)
            output = np.array(output) if isinstance(output, tf.Tensor) else output
            output[np.arange(output.shape[0]), replay_action.flatten()] = y_i_array.flatten()
            output = tf.convert_to_tensor(output)
            loss = tf.reduce_mean(tf.square(output - model_target(state)))
        grads = tape.gradient(loss, model_target.trainable_variables)
        model_target.optimizer.apply_gradients(zip(grads, model_target.trainable_variables))

        self.params.avg_loss += loss
        if iteration % self.params.update_target_every == self.params.update_target_every - 1:
            self.models.update_target_models()

    def train_agent(self):
        # set hyper parameters
        epochs = self.params.epochs
        max_iteration = self.params.max_iteration
        greedy_epsilon_decay = self.params.greedy_epsilon_decay


        # start training
        for episode_num in range(epochs):
            state = self.reset_env()
            for iteration in range(max_iteration):
                action = self.sample_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.params.add_reward(reward)
                next_state = np.reshape(next_state, (1, 4))
                if done:
                    self.replay_q.add(state, action, (reward - 100), next_state, done)
                    replays = self.replay_q.sample_batch()
                    if self.params.who_is_training:
                        self.update_models(self.models.Qnet_values_A, self.models.Qnet_target_B, replays, iteration)
                    else:
                        self.update_models(self.models.Qnet_values_B, self.models.Qnet_target_A, replays, iteration)
                    break
                else:
                    self.replay_q.add(state, action, reward, next_state, done)
                    replays = self.replay_q.sample_batch()
                    if self.params.who_is_training:
                        self.update_models(self.models.Qnet_values_A, self.models.Qnet_target_B, replays, iteration)
                    else:
                        self.update_models(self.models.Qnet_values_B, self.models.Qnet_target_A, replays, iteration)
                state = next_state
            self.params.update_greedy_epsilon(greedy_epsilon_decay)
            self.params.update_avg_reward()
            print("Episode: {}, Loss: {:.6f}, Epsilon: {:.6f}, accumulate_reward :{:.2f}, avg_reward: {:.2f}".
                  format(episode_num, self.params.avg_loss / self.params.max_iteration, self.params.greedy_epsilon,
                         self.params.accumulate_reward,self.params.avg_reward))
            if episode_num % 25 == 0:
                self.models.save_weights()


if __name__ == "__main__":
    agent = Agent()
    agent.train_agent()

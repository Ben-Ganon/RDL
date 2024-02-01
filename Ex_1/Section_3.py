import gymnasium as gym
import numpy as np

from Params import Params
import random
from collections import deque
from QNet import QNet
import tensorflow as tf
import time

INPUT_SIZE = 4
OUT_PUT_SIZE = 2
OPTIMIZER = "sgd"
LEARNING_RATE = 0.01
LAYER_SIZES = [32, 32, 32]
MAX_ITERATION = 50000
DISCOUNT = 0.93
EPOCHS = 500
N_REPLAY_SIZE = 1000
BATCH_SIZE = 32
GREEDY_EPSILON = 0.9
GREEDY_EPSILON_DECAY = 0.6
RENDER_MODE = False
REWARD_QUE_SIZE = 100
ENV_NAME = "CartPole-v1"
MIN_GREEDY_EPSILON = 0.0
LOAD = False
UPDATE_TARGET_EVERY = 10


class Q_function():
    def __init__(self, load, layers, output_size, optimizer, learning_rate):
        self.Qnet_target_A = QNet(layer_sizes=layers, output_size=output_size, optimizer=optimizer,
                                  learning_rate=learning_rate)
        self.Qnet_target_B = QNet(layer_sizes=layers, output_size=output_size, optimizer=optimizer,
                                  learning_rate=learning_rate)
        self.Qnet_values_A = QNet(layer_sizes=layers, output_size=output_size, optimizer=optimizer,
                                  learning_rate=learning_rate)
        self.Qnet_values_B = QNet(layer_sizes=layers, output_size=output_size, optimizer=optimizer,
                                  learning_rate=learning_rate)
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

    def update_target_models(self, model_target, who_is_training):
        dummy_input = tf.keras.Input(shape=INPUT_SIZE)
        self.Qnet_target_A(dummy_input)
        self.Qnet_target_B(dummy_input)
        if who_is_training:
            self.Qnet_target_A.set_weights(model_target.get_weights())
            self.Qnet_values_A.set_weights(model_target.get_weights())
        else:
            self.Qnet_target_B.set_weights(model_target.get_weights())
            self.Qnet_values_B.set_weights(model_target.get_weights())

    def save_weights(self):
        self.Qnet_target_A.save_weights("Weights/Qnet_target_A.weights")
        self.Qnet_target_B.save_weights("Weights/Qnet_target_B.weights")
        self.Qnet_values_A.save_weights("Weights/Qnet_values_A.weights")
        self.Qnet_values_B.save_weights("Weights/Qnet_values_B.weights")


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

    def add(self, state, action, reward, next_state, next_action, q_values, done):
        self.que.append((state, action, reward, next_state, next_action, q_values, done))

    def get_batch_new(self, replays, model_values):
        replay_state = []
        replay_action = []
        replay_reward = []
        replay_next_step = []
        replay_next_action = []
        replay_q_values = []
        y_i_array = []
        for replay in replays:
            replay_state.append(replay[0][0])
            replay_action.append(replay[1])
            replay_reward.append(replay[2])
            replay_next_step.append(replay[3])
            replay_next_action.append(replay[4])
            replay_q_values.append(replay[5])
            y_i = replay[2] if replay[6] else replay[2] + DISCOUNT * (
                model_values.call(tf.convert_to_tensor(replay[3]))[0][replay[4]])
            y_i_array.append(y_i)
        replay_state = np.array(replay_state).reshape(len(replays), 4)
        replay_action = np.array(replay_action).reshape(len(replays), 1)
        replay_reward = np.array(replay_reward).reshape(len(replays), 1)
        replay_next_step = np.array(replay_next_step).reshape(len(replays), 4)
        y_i_array = np.array(y_i_array).reshape(len(replays), 1)
        return replay_state, replay_action, replay_reward, replay_next_step, y_i_array, replay_q_values, replay_next_action


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
        self.who_is_training = True  # 1 for A, 0 for B
        self.print_params()
        self.count_A = 0
        self.count_B = 0
        pass

    def print_params(self):
        print(
            "greedy_epsilon: {}, greedy_epsilon_decay: {}, max_iteration: {}, discount: {}, update_target_every: {}, epochs: {}".
            format(self.greedy_epsilon, self.greedy_epsilon_decay, self.max_iteration, self.discount,
                   self.update_target_every, self.epochs))

    def update_avg_reward(self):
        self.reward_que.append(self.accumulate_reward)
        self.avg_reward = np.mean(self.reward_que)

    def reset_params(self):
        self.accumulate_reward = 0
        self.avg_loss = 0
        self.done = False
        self.count_A = 0
        self.count_B = 0

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
        self.replay_q_A = replay(n_replay_size=N_REPLAY_SIZE, batch_size=BATCH_SIZE)
        self.replay_q_B = replay(n_replay_size=N_REPLAY_SIZE, batch_size=BATCH_SIZE)
        # init Qnets
        self.models = Q_function(load=LOAD, layers=LAYER_SIZES, output_size=self.env.action_space.n,
                                 optimizer=OPTIMIZER,
                                 learning_rate=LEARNING_RATE)
        # init params
        self.params = Params(greedy_epsilon=GREEDY_EPSILON, greedy_epsilon_decay=GREEDY_EPSILON_DECAY,
                             max_iteration=MAX_ITERATION, discount=DISCOUNT,
                             update_target_every=UPDATE_TARGET_EVERY, epochs=EPOCHS)

    def reset_env(self):
        state = self.env.reset()[0]
        state = np.reshape(state, (1, 4))
        self.params.reset_params()
        if self.env.render_mode:
            self.env.render()

        return state

    def sample_action(self, state, model_target=None):
        rand = np.random.random()
        if rand < self.params.greedy_epsilon:
            # self.params.who_is_training = not self.params.who_is_training
            # q_values = model_target.call(tf.convert_to_tensor(state))
            action = np.random.choice([0, 1])
            # tensor that contains 1 where the action was taken and 0 otherwise
            q_values = np.zeros((1, 2))
            # q_values[0][action] = 1
            return action, q_values
        else:
            q_values = model_target.call(tf.convert_to_tensor(state))
            action = np.argmax(q_values)
            return action, q_values

    def update_models(self, model_target, model_to_update, model_values, replays, queue, iterations):
        replay_state, replay_action, replay_reward, replay_next_step, y_i_array, replay_q_values, replay_next_action = queue.get_batch_new(
            replays, model_values)
        # get the value after call only for the action taken
        state = tf.convert_to_tensor(replay_state)

        # output = np.reshape(replay_q_values, (len(replays), 2))
        # output[range(len(replays)), replay_action[:, 0]] = y_i_array[:, 0]
        # history = model_target.fit(state, output, verbose=0)
        # self.params.avg_loss += history.history['loss'][0]
        with tf.GradientTape() as tape:
            output = model_target(state)
            predictions = tf.gather_nd(output, replay_action, batch_dims=1)
            loss = model_target.compiled_loss(tf.constant(y_i_array, shape=(len(replays), 1)), predictions)
        gradients = tape.gradient(loss, model_target.trainable_variables)
        model_target.optimizer.apply_gradients(zip(gradients, model_target.trainable_variables))
        self.params.avg_loss += loss
        if iterations % UPDATE_TARGET_EVERY == 0:
            model_to_update.set_weights(model_target.get_weights())

    def train_agent(self):
        # set hyper parameters
        epochs = self.params.epochs
        max_iteration = self.params.max_iteration
        greedy_epsilon_decay = self.params.greedy_epsilon_decay
        model_target_A = self.models.Qnet_target_A
        model_target_B = self.models.Qnet_target_B
        model_values_A = self.models.Qnet_values_A
        model_values_B = self.models.Qnet_values_B
        file_name = time.time()

        for episode_num in range(epochs):
            count_iteration = 0
            done = False
            state = self.reset_env()

            rand = np.random.random()
            if rand < 0.5:
                self.params.who_is_training = True
                model_target = model_target_A
                model_values = model_target_B
                model_to_train = model_values_A
                queue = self.replay_q_A
                self.params.count_A += 1
            else:
                self.params.who_is_training = False
                model_target = model_target_B
                model_values = model_target_A
                model_to_train = model_values_B
                queue = self.replay_q_B
                self.params.count_B += 1

            while not done:
                count_iteration += 1

                action, q_values = self.sample_action(state, model_target=model_target)

                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                self.params.add_reward(reward)
                next_state = np.reshape(next_state, (1, 4))
                next_action = np.argmax(model_target.call(tf.convert_to_tensor(next_state)))
                queue.add(state, action, reward, next_state, next_action, q_values, done)
                replays = queue.sample_batch()
                # self.update_models(model_to_train, model_target, model_values, replays, queue, count_iteration)

                replay_state, replay_action, replay_reward, replay_next_step, y_i_array, replay_q_values, replay_next_action = queue.get_batch_new(
                    replays, model_values)
                # get the value after call only for the action taken
                state_update = tf.convert_to_tensor(replay_state)

                # output = np.reshape(replay_q_values, (len(replays), 2))
                # output[range(len(replays)), replay_action[:, 0]] = y_i_array[:, 0]
                # history = model_target.fit(state, output, verbose=0)
                # self.params.avg_loss += history.history['loss'][0]
                with tf.GradientTape() as tape:
                    output = model_to_train(state_update)
                    predictions = tf.gather_nd(output, replay_action, batch_dims=1)
                    loss = model_to_train.compiled_loss(tf.constant(y_i_array, shape=(len(replays), 1)), predictions)
                gradients = tape.gradient(loss, model_to_train.trainable_variables)
                model_to_train.optimizer.apply_gradients(zip(gradients, model_to_train.trainable_variables))
                self.params.avg_loss += loss
                if count_iteration % UPDATE_TARGET_EVERY == 0:
                    model_target.set_weights(model_to_train.get_weights())
                state = next_state

            self.params.update_greedy_epsilon(greedy_epsilon_decay)
            self.params.update_avg_reward()
            print(
                "Episode: {}, Loss: {:.6f}, Epsilon: {:.6f}, accumulate_reward :{:.2f}, avg_reward: {:.2f}, count A: {}, count B: {}".
                format(episode_num, self.params.avg_loss / (count_iteration * BATCH_SIZE), self.params.greedy_epsilon,
                       self.params.accumulate_reward, self.params.avg_reward, self.params.count_A, self.params.count_B))
            with open(f'Ex_1/runs/{file_name}_log.txt', "a") as f:
                f.write(
                    "Episode: {}, Loss: {:.6f}, accumulate_reward :{:.2f}, avg_reward: {:.2f} \n".
                    format(episode_num, self.params.avg_loss / (count_iteration * BATCH_SIZE),
                           self.params.accumulate_reward, self.params.avg_reward)
                )
            if episode_num % 25 == 0:
                self.models.save_weights()


if __name__ == "__main__":
    agent = Agent()
    agent.train_agent()

import gymnasium as gym
import numpy as np

from Params import Params
import random
from collections import deque
from QNet import QNet
import tensorflow as tf

N_REPLAY_SIZE = 5000
EPOCHS = 1000
LEARNING_RATE = 0.01
LAYER_NUM = 3
LAYER_SIZE = 64
LAYER_SIZE_ARRAY = [LAYER_SIZE] * LAYER_NUM
GREEDY_EPSILON = 0.5
GREEDY_EPSILON_DECAY = 0.999
MAX_ITERATION = 5000
DISCOUNT = 0.9
BATCH_SIZE = 1
UPDATE_TARGET_EVERY = 500
DO_RENDER = True
LOAD = False
RENDER_METHOD = "human" if DO_RENDER else None


def initialize_experience_replay():
    que = deque(maxlen=N_REPLAY_SIZE)
    return que


def sample_batch(que, batch_size):
    if len(que) < batch_size:
        return random.sample(que, len(que))
    return random.sample(que, batch_size)

def LOAD_WEIGHTS(Qnet_target_A, Qnet_target_B, Qnet_values_A, Qnet_values_B):
    dummy_input = tf.keras.Input(shape=4)
    Qnet_target_A(dummy_input)
    Qnet_target_B(dummy_input)
    Qnet_values_A(dummy_input)
    Qnet_values_B(dummy_input)
    Qnet_target_A.load_weights("Qnet_target_A.weights.h5")
    Qnet_target_B.load_weights("Qnet_target_B.weights.h5")
    Qnet_values_A.load_weights("Qnet_values_A.weights.h5")
    Qnet_values_B.load_weights("Qnet_values_B.weights.h5")


def train_agent():
    update_flag = True
    env = gym.make('CartPole-v1', render_mode=RENDER_METHOD)
    observation = env.observation_space
    action = env.action_space
    replay_q = initialize_experience_replay()
    greedy_epsilon = GREEDY_EPSILON
    Qnet_target_A = QNet(layer_sizes=[24, 24, 24], output_size=action.n, optimizer="adam", learning_rate=LEARNING_RATE)
    Qnet_target_B = QNet(layer_sizes=[24, 24, 24], output_size=action.n, optimizer="adam", learning_rate=LEARNING_RATE)
    Qnet_values_A = QNet(layer_sizes=[24, 24, 24], output_size=action.n, optimizer="adam", learning_rate=LEARNING_RATE)
    Qnet_values_B = QNet(layer_sizes=[24, 24, 24], output_size=action.n, optimizer="adam", learning_rate=LEARNING_RATE)

    if LOAD:
        LOAD_WEIGHTS(Qnet_target_A, Qnet_target_B, Qnet_values_A, Qnet_values_B)

    for episode_num in range(EPOCHS):
        state = env.reset()[0]
        state = np.reshape(state, (1, 4))
        done = False
        if DO_RENDER:
            env.render()
        avg_loss = 0
        for iteration in range(MAX_ITERATION):
            rand = np.random.random()
            greedy_epsilon *= GREEDY_EPSILON_DECAY
            if rand < GREEDY_EPSILON:
                action = np.random.choice([0, 1])
                update_flag = not update_flag
            else:
                rand = np.random.random()
                if rand < 0.5:
                    action = np.argmax(Qnet_values_A.call(state))
                    update_flag = True
                else:
                    action = np.argmax(Qnet_values_B.call(state))
                    update_flag = False
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, (1, 4))
            replay_q.append((state, action, reward, next_state))
            replays = sample_batch(replay_q, BATCH_SIZE)
            replay_state = np.array([replay[0] for replay in replays]).reshape(len(replays), 4)
            replay_action= np.array([replay[1] for replay in replays]).reshape(len(replays), 1)
            replay_reward = np.array([replay[2] for replay in replays]).reshape(len(replays), 1)
            replay_next_step = np.array([replay[3] for replay in replays]).reshape(len(replays), 4)
            if done:
                y_i = reward
            else:
                if update_flag:
                    y_i = replay_reward + DISCOUNT * Qnet_values_B.call(replay_next_step)
                    loss = Qnet_values_A.custom_train_step(replay_state, y_i)
                    avg_loss += loss
                    if iteration % UPDATE_TARGET_EVERY == UPDATE_TARGET_EVERY - 1:
                        # copy model parameters from Qnet_values to Qnet_target
                        Qnet_target_A.set_weights(Qnet_values_A.get_weights())
                else:
                    y_i = replay_reward + DISCOUNT * Qnet_values_A.call(replay_next_step)
                    loss = Qnet_values_B.custom_train_step(replay_state, y_i)
                    avg_loss += loss
                    if iteration % UPDATE_TARGET_EVERY == UPDATE_TARGET_EVERY - 1:
                        # copy model parameters from Qnet_values to Qnet_target
                        Qnet_target_B.set_weights(Qnet_values_B.get_weights())
            state = next_state
            if done:
                break
        print("Episode: {}, Loss: {}, Epsilon: {}".
              format(episode_num, avg_loss / MAX_ITERATION, greedy_epsilon))
    Qnet_target_A.save_weights("Qnet_target_A.weights.h5")
    Qnet_target_B.save_weights("Qnet_target_B.weights.h5")
    Qnet_values_A.save_weights("Qnet_values_A.weights.h5")
    Qnet_values_B.save_weights("Qnet_values_B.weights.h5")


if __name__ == "__main__":
    train_agent()

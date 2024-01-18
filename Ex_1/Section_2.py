import gymnasium as gym
import numpy as np

from Params import Params
import random
from collections import deque
from QNet import QNet
N_REPLAY_SIZE = 5000
EPOCHS = 5000
LEARNING_RATE = 0.01
LAYER_NUM = 3
LAYER_SIZE = 64
LAYER_SIZE_ARRAY = [LAYER_SIZE] * LAYER_NUM
GREEDY_EPSILON = 0.5
GREEDY_EPSILON_DECAY = 0.999
MAX_ITERATION = 100
DISCOUNT = 0.9
BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 50
DO_RENDER = True
LOAD = True
RENDER_METHOD = "human" if DO_RENDER else None


def initialize_experience_replay():
    que = deque(maxlen=N_REPLAY_SIZE)
    return que

def sample_batch(que, batch_size):
    if len(que) < batch_size:
        return random.sample(que, len(que))
    return random.sample(que, batch_size)


def train_agent():
    env = gym.make('CartPole-v1', render_mode=RENDER_METHOD)
    observation = env.observation_space
    action = env.action_space
    replay_q = initialize_experience_replay()
    greedy_epsilon = GREEDY_EPSILON
    Qnet_target = QNet(layer_sizes=[24, 24, 24], output_size=action.n, optimizer="adam", learning_rate=LEARNING_RATE)
    Qnet_values = QNet(layer_sizes=[24, 24, 24], output_size=action.n, optimizer="adam", learning_rate=LEARNING_RATE)
    if LOAD:
        Qnet_target.load_weights("Qnet_target.weights.h5")
        Qnet_values.load_weights("Qnet_values.weights.h5")
    dummy_outputs = Qnet_target.call(np.zeros((1, 4)))
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
            else:
                action = np.argmax(Qnet_values.call(state))
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, (1, 4))
            replay_q.append((state, action, reward, next_state))
            replays = sample_batch(replay_q, BATCH_SIZE)
            replay_array = np.array([replay[0] for replay in replays]).reshape(len(replays), 4)
            reward_array = np.array([replay[2] for replay in replays]).reshape(len(replays), 1)
            Qnet_outputs = Qnet_values.call(replay_array)
            if done:
                y_i = reward
            else:
                y_i = reward + DISCOUNT * np.max(Qnet_outputs, axis=1)
            loss = Qnet_values.custom_train_step(replay_array, y_i)
            avg_loss += loss
            if iteration % UPDATE_TARGET_EVERY == UPDATE_TARGET_EVERY -1:
                # copy model parameters from Qnet_values to Qnet_target
                Qnet_target.set_weights(Qnet_values.get_weights())
            state = next_state
            if done:
                break
        print("Episode: {}, Loss: {}, Epsilon: {}".
              format(episode_num, avg_loss / MAX_ITERATION, greedy_epsilon))
    Qnet_target.save_weights("Qnet_target.weights.h5")
    Qnet_values.save_weights("Qnet_values.weights.h5")

if __name__ == "__main__":
    train_agent()
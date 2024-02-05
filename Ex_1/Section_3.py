import gymnasium as gym
import numpy as np
import tensorflow as tf
import random
from collections import deque
from QNet import QNet, custom_train_step
import csv
import sys
N_REPLAY_SIZE = 10000
RANDOM_REPLAY_SIZE = 1024
EPISODES = 1000
LEARNING_RATE = 0.0007
LAYER_NUM = 3
LAYER_SIZE = 16
# # Layer size array where first layer size is LAYER_SIZE and each subsequent layer size is current_layer_size/2
# LAYER_SIZE_ARRAY = [LAYER_SIZE // (2 ** i) for i in range(LAYER_NUM)]
LAYER_SIZE_ARRAY = [32, 32, 32]
# LAYER_SIZE_ARRAY = [LAYER_SIZE for i in range(LAYER_NUM)]
GREEDY_EPSILON = 1.0
GREEDY_EPSILON_DECAY = 0.9
MIN_GREEDY_EPSILON = 0.01
MAX_ITERATION = 3000
DISCOUNT = 0.95
BATCH_SIZE = 32
RANDOM_BATCH_SIZE = 16
UPDATE_TARGET_EVERY = 8
REWARD_ITER = 100
OPTIMIZER = "adam"
DO_RENDER = False
LOAD = False
RENDER_METHOD = "human" if DO_RENDER else None
A_CHECKPT = "Ex_1/checkpoints/Qnet_A_checkpoint_final"
B_CHECKPT = "Ex_1/checkpoints/Qnet_B_checkpoint_final"

train_summary_writer = tf.summary.create_file_writer("Ex_1/section_3_logs/final")
tf.random.set_seed(42)


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
    replay_q_a = initialize_experience_replay()
    replay_q_b = initialize_experience_replay()
    replay_random_a = initialize_experience_replay()
    replay_random_b = initialize_experience_replay()
    greedy_epsilon = GREEDY_EPSILON
    Qnet_A = QNet(layer_sizes=LAYER_SIZE_ARRAY, output_size=action.n, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE)
    Qnet_B = QNet(layer_sizes=LAYER_SIZE_ARRAY, output_size=action.n, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE)

    print(f"paramters: {LAYER_SIZE_ARRAY}\noptimizer: {OPTIMIZER}\nlearning rate: {LEARNING_RATE}\n"
          f"greedy epsilon: {GREEDY_EPSILON}\ngreedy epsilon decay: {GREEDY_EPSILON_DECAY}\n"
          f"max iterations: {MAX_ITERATION}\ndiscount: {DISCOUNT}\nbatch size: {BATCH_SIZE}\n"
          f"update target every: {UPDATE_TARGET_EVERY}\nreward iteration: {REWARD_ITER}\n"
          f"Replay size: {N_REPLAY_SIZE}\n")
    avg_rewards = []
    not_updated_count = 0
    for episode_num in range(EPISODES):
        state = env.reset()[0]
        # state = np.reshape(state, (1, 4))
        state = tf.constant(np.reshape(state, (1, 4)))
        done = False
        if DO_RENDER:
            env.render()
        avg_loss = 0
        if episode_num % REWARD_ITER == REWARD_ITER - 1:
            # avg_reward /= REWARD_ITER
            # print("Average reward for last {} episodes: {}".format(REWARD_ITER, avg_reward))
            # avg_reward = 0
            Qnet_A.save_weights(A_CHECKPT)
            Qnet_B.save_weights(B_CHECKPT)
        iter_reward = 0
        for iteration in range(MAX_ITERATION):
            random_network = np.random.choice(['a', 'b'])
            rand = np.random.random()
            if rand < greedy_epsilon:
                action = np.random.choice([0, 1])
            else:
                if random_network == 'a':
                    action = np.argmax(Qnet_A.call(state))
                elif random_network == 'b':
                    action = np.argmax(Qnet_B.call(state))
            next_state, reward, done, truncated, _ = env.step(action)
            iter_reward += reward
            # next_state = np.reshape(next_state, (1, 4))
            next_state = tf.constant(np.reshape(next_state, (1, 4)))
            if random_network == 'a':
                next_action = np.argmax(Qnet_A.call(next_state))
                replay_q_a.append((state, action, reward, next_state, next_action, done))
                if greedy_epsilon < RANDOM_REPLAY_SIZE:
                    replay_random_a.append((state, action, reward, next_state, next_action, done))
                replays = sample_batch(replay_q_a, BATCH_SIZE)
                random_replays = sample_batch(replay_random_a, RANDOM_BATCH_SIZE)
                replays += random_replays
            elif random_network == 'b':
                next_action = np.argmax(Qnet_B.call(next_state))
                replay_q_b.append((state, action, reward, next_state, next_action, done))
                if greedy_epsilon < RANDOM_REPLAY_SIZE:
                    replay_random_b.append((state, action, reward, next_state, next_action, done))
                replays = sample_batch(replay_q_b, BATCH_SIZE)
                random_replays = sample_batch(replay_random_b, RANDOM_BATCH_SIZE)
                replays += random_replays
            replay_length = len(replays)
            y_i_array = []
            state_array = []
            action_array = []
            # add the replay values to the array in the most efficient way
            for replay in replays:
                if replay[5]:
                    y_i_array.append(replay[2])
                else:
                    if random_network == 'a':
                        y_i_array.append(replay[2] + DISCOUNT * Qnet_B.call(replay[3])[0][replay[4]].numpy())
                    elif random_network == 'b':
                        y_i_array.append(replay[2] + DISCOUNT * Qnet_A.call(replay[3])[0][replay[4]].numpy())
                state_array.append(replay[0])
                action_array.append(replay[1])
            y_i_array = tf.constant(y_i_array, shape=(replay_length, 1))
            # state_array = tf.constant(state_array, shape=(replay_length, 4))
            state_array = tf.concat(state_array, axis=0)
            action_array = tf.constant(action_array, shape=(replay_length, 1))
            if random_network == 'a':
                loss = custom_train_step(Qnet_A, state_array, y_i_array, action_array)
            elif random_network == 'b':
                loss = custom_train_step(Qnet_B, state_array, y_i_array, action_array)
            avg_loss += loss
            state = next_state
            iterations_done = iteration
            if done or truncated:
                break
        greedy_epsilon = max(MIN_GREEDY_EPSILON, greedy_epsilon * GREEDY_EPSILON_DECAY)
        avg_rewards.append(iter_reward)
        with train_summary_writer.as_default():
            tf.summary.scalar('avg_loss', avg_loss / iterations_done, step=episode_num)
        with train_summary_writer.as_default():
            tf.summary.scalar('rewards_per_episode', iter_reward, step=episode_num)

        if len(avg_rewards) > 100:
            avg_rewards.pop(0)
        print(f"Episode: {episode_num}, Loss: {(avg_loss / (iterations_done)):4.6f}., Epsilon: {greedy_epsilon:1.3f}, "
              f"Reward: {iter_reward}, Average Reward: {sum(avg_rewards) / len(avg_rewards):3.0f}")
    Qnet_A.save_weights(A_CHECKPT)
    Qnet_B.save_weights(B_CHECKPT)
    env.close()

def test_agent():
    env = gym.make('CartPole-v1', render_mode=RENDER_METHOD)
    observation = env.observation_space
    action = env.action_space
    replay_q = initialize_experience_replay()
    greedy_epsilon = GREEDY_EPSILON
    Qnet_target = QNet(layer_sizes=LAYER_SIZE_ARRAY, output_size=action.n, optimizer=OPTIMIZER,
                       learning_rate=LEARNING_RATE)
    Qnet_values = QNet(layer_sizes=LAYER_SIZE_ARRAY, output_size=action.n, optimizer=OPTIMIZER,
                       learning_rate=LEARNING_RATE)
    dummy_outputs = Qnet_target.call(np.zeros((1, 4)))
    dummy_outputs = Qnet_values.call(np.zeros((1, 4)))
    Qnet_target.load_weights(A_CHECKPT)
    avg_reward = 0
    for episode_num in range(EPISODES):
        episode_reward = 0
        state = env.reset()[0]
        state = tf.constant(np.reshape(state, (1, 4)))
        if DO_RENDER:
            env.render()
        if episode_num % REWARD_ITER == REWARD_ITER - 1:
            avg_reward /= REWARD_ITER
            print("Average reward for last {} episodes: {}".format(REWARD_ITER, avg_reward))
            avg_reward = 0
        for iteration in range(MAX_ITERATION):
            action = np.argmax(Qnet_target.call(state))
            next_state, reward, done, truncated, _ = env.step(action)
            iterations_done = iteration
            if done or truncated:
                break
            avg_reward += reward
            episode_reward += reward
            next_state = tf.constant(np.reshape(next_state, (1, 4)))
            state = next_state
        print(f'Episode: {episode_num}, Reward: {episode_reward}, Iterations: {iterations_done}')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_agent()
        elif sys.argv[1] == "test":
            test_agent()
        else:
            raise ValueError("Invalid argument. Please use either 'train' or 'test' as the argument.")

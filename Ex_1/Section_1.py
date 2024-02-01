import gymnasium as gym
import numpy as np
from Params import Params
import tensorflow as tf
import sys
ALPHA = 0.9
DISCOUNT = 0.9
GREEDY_EPSILON = 0.5
GREEDY_MIN = 0.1
EPOCHS = 5000
RENDER = False
DONE = False
RENDER_MODE = "human" if RENDER else None
GREEDY_DECAY = 0.001 if DONE else 0.999
train_summary_writer = tf.summary.create_file_writer("Ex_1/section_1/section_1_logs")

simple_map = ["SFFF", "FFHF", "FFFF", "HFFG"]

rewards_per_episode = []
avg_num_steps_to_goal = []

def get_next_state(state, action):
    row = state // 4
    col = state % 4
    if action == 0:
        row = max(row - 1, 0)
    elif action == 1:
        row = min(row + 1, 3)
    elif action == 2:
        col = max(col - 1, 0)
    elif action == 3:
        col = min(col + 1, 3)
    return row * 4 + col

def Qtable_checkpoint(q_lookup, i, max_epochs):
    if i == 500:
        np.save("Ex_1/section_1/q_lookup_500.npy", q_lookup)
        np.savetxt("Ex_1/section_1/q_lookup_500.csv", q_lookup, fmt="%0.3f", delimiter=",")
    elif i == 2000:
        np.save("Ex_1/section_1/q_lookup_2000.npy", q_lookup)
        np.savetxt("Ex_1/section_1/q_lookup_2000.csv", q_lookup, fmt="%0.3f", delimiter=",")
    elif i == max_epochs - 1:
        np.save("Ex_1/section_1/q_lookup_final.npy", q_lookup)
        np.savetxt("Ex_1/section_1/q_lookup_final.csv", q_lookup, fmt="%0.3f", delimiter=",")

def train_agent():
    env = gym.make('FrozenLake-v1', desc=simple_map, map_name="4x4", is_slippery=True, render_mode=RENDER_MODE)
    observation = env.observation_space
    action = env.action_space
    q_lookup = np.zeros((env.observation_space.n, env.action_space.n))
    # q_lookup = np.load("q_lookup.npy")
    params = Params(alpha=ALPHA, discount=DISCOUNT, greedy_min=GREEDY_MIN, greedy_decay=GREEDY_DECAY, greedy_epsilon=GREEDY_EPSILON, epochs=EPOCHS)
    print("Params: ", str(params))
    steps_total = 0
    for i in range(params.epochs):
        state = env.reset()[0]
        if RENDER:
            env.render()
        total_reward = 0

        if i % 100 == 99:
            steps_average = steps_total / 100
            avg_num_steps_to_goal.append(steps_average)
            with train_summary_writer.as_default():
                tf.summary.scalar('avg_steps_to_goal', steps_average, step=i)
            steps_total = 0
        for j in range(100):
        # print(f"Episode: {i}, Step: {j} action: {action}, greedy_factor: {params.greedy_epsilon}")

            # make an epsilon greedy action
            rand = np.random.random()
            if rand < params.greedy_epsilon:
                action = np.random.choice([0, 1, 2, 3])
            else:
                action = np.argmax(q_lookup[state, :])
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                target = reward
                total_reward += reward
                params.decay_greedy()
                q_lookup[state, action] = (1 - params.alpha) * q_lookup[state, action] + params.alpha * target
                # add number of steps left to goal in this episode to steps_total
                if reward == 1:
                    steps_total += j
                else:
                    steps_total += 100
                break
            else:
                next_action = np.argmax(q_lookup[next_state, :])
                target = reward + params.discount * np.max(q_lookup[next_state, next_action])
                params.decay_greedy()
                q_lookup[state, action] = (1 - params.alpha) * q_lookup[state, action] + params.alpha * target
                state = next_state
            print(f"Episode: {i}, Step:{j}, Action:{action}, Greedy:{params.greedy_epsilon}, ")
        Qtable_checkpoint(q_lookup, i, params.epochs)
        # steps_total += 15 - state
        rewards_per_episode.append(total_reward)
        with train_summary_writer.as_default():
            tf.summary.scalar('reward', total_reward, step=i)

def test_agent():
    env = gym.make('FrozenLake-v1', desc=simple_map, map_name="4x4", is_slippery=True, render_mode=RENDER_MODE)
    observation = env.observation_space
    action = env.action_space
    q_lookup = np.load("Ex_1/section_1/q_lookup_final.npy")
    total_reward = 0
    for i in range(100):
        state = env.reset()[0]
        if RENDER:
            env.render()
        for j in range(100):
            action = np.argmax(q_lookup[state, :])
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                total_reward += reward
                break
            else:
                state = next_state
    print(f"Average Reward: {total_reward / 100}")
    env.close()


    np.save("Ex_1/section_1/q_lookup.npy", q_lookup)
    np.savetxt("Ex_1/section_1/q_lookup.csv", q_lookup, fmt="%0.3f",delimiter=",")
    with open("Ex_1/section_1/rewards_per_episode.txt", "w") as f:
        f.write(str(rewards_per_episode))
    with open("Ex_1/section_1/avg_num_steps_to_goal.txt", "w") as f:
        f.write(str(avg_num_steps_to_goal))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_agent()
        elif sys.argv[1] == "test":
            test_agent()


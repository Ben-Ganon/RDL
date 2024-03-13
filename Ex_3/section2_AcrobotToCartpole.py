import random

import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
import csv

# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

# env = gym.make('CartPole-v1')
render = False
render_mode = "human" if render else None
env = gym.make('CartPole-v1', render_mode=render_mode)
# env = gym.make('Acrobot-v1', render_mode=render_mode)
# env = gym.make('MountainCarContinuous-v0', render_mode=render_mode)
np.random.seed(42)
tf.set_random_seed(42)
Restore_name = 'Acrobot-v1'
policy_layer = 12
value_layer = 12


class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.target = tf.placeholder(tf.float32, name="target")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, value_layer], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [value_layer], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [value_layer, value_layer], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [value_layer], initializer=tf2_initializer)
            self.W3 = tf.get_variable("W3", [value_layer, 1], initializer=tf2_initializer)
            self.b3 = tf.get_variable("b3", [1], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.loss = tf.reduce_mean(tf.square(self.target - self.output))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, policy_layer], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [policy_layer], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [policy_layer, policy_layer], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [policy_layer], initializer=tf2_initializer)
            self.W3 = tf.get_variable("W3", [policy_layer, self.action_size], initializer=tf2_initializer)
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def state_zero_padding(state, env):
    if env.spec.id == "MountainCarContinuous-v0":
        state = np.append(state, [0, 0, 0, 0])
    elif env.spec.id == "CartPole-v1":
        state = np.append(state, [0, 0])
    state = np.reshape(state, [1, 6])
    return state


def action_zero_padding(action, env):
    if env.spec.id == "MountainCarContinuous-v0":
        action = np.append(action, [0, 0])
    elif env.spec.id == "CartPole-v1":
        action = np.append(action, 0)
    return action


def delete_output_padding(state, env):
    if env.spec.id == "MountainCarContinuous-v0":
        state = state[0]
    elif env.spec.id == "CartPole-v1":
        state = state[0:2]
        state /= np.sum(state)
    return state


def reward_mountain_car(state, action, next_State):
    position = state[0][0]
    velocity = state[0][1]
    goal_threshold = 0.45  # This should be the x-position of the flag.

    # Compute the distance to the flag from the current position
    distance_to_flag = goal_threshold - position

    # Negative reward proportional to the square of action magnitude to discourage large actions
    action_penalty = -0.1 * (action ** 2)

    # Large reward for reaching the goal
    reward = 100 if position >= goal_threshold else 0

    # Reward for correct direction velocity
    if position < goal_threshold:
        reward += velocity if distance_to_flag > 0 else -velocity

    # Combine reward with action penalty and a time penalty
    reward += action_penalty - 1

    if position == next_State[0][0]:
        reward -= 1

    return reward


def run():
    # Define hyperparameters
    state_size = env.observation_space.shape[0]
    if env.spec.id != "MountainCarContinuous-v0":
        action_size = env.action_space.n
    else:
        action_size = env.action_space.shape[0]
    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0004
    learning_rate_baseline = 0.0008
    render = False
    mountain_car_decay = 0.99

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(6, 3, learning_rate)
    baseline = ValueNetwork(6, learning_rate_baseline)
    # Start training the agent with REINFORCE algorithm
    saver = tf.train.Saver()
    output_layer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_network/W3') + \
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_network/b3')
    with tf.Session() as sess:
        saver.restore(sess, f"/Users/Administrator/PycharmProjects/RDL_ex3/Ex_3/{Restore_name}/{Restore_name}.ckpt")
        sess.run(tf.variables_initializer(output_layer_vars))
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "value"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        avg_reward_history = []

        for episode in range(max_episodes):
            state = env.reset()[0]
            zero_position = state[0]
            episode_transitions = []

            for step in range(max_steps):
                state = state_zero_padding(state, env)
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                actions_distribution = delete_output_padding(actions_distribution, env)
                if type(actions_distribution) is np.float32:
                    rand = np.random.rand()
                    mountain_car_decay *= 0.99
                    if mountain_car_decay > rand:
                        rand = 2 * np.random.rand() - 1
                        action = np.array([random.choice([rand, actions_distribution])])
                    else:
                        action = np.array([actions_distribution])
                else:
                    action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if env.spec.id == "MountainCarContinuous-v0":
                    reward = reward_mountain_car(state,action, next_state)

                if render:
                    env.render()
                value = sess.run(baseline.output, {baseline.state: state})
                if env.spec.id != "MountainCarContinuous-v0":
                    action_one_hot = np.zeros(action_size)
                    action_one_hot[action] = 1
                else:
                    action_one_hot = action
                episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done,
                               value=value))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                        avg_reward_history.append(average_rewards)
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                     round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Compute Rt for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in
                                              enumerate(episode_transitions[t:])) - transition.value  # Rt - V(s)
                transition_action = action_zero_padding(transition.action, env)
                feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return,
                             policy.action: transition_action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                feed_dict = {baseline.state: transition.state, baseline.target: total_discounted_return}
                _, loss = sess.run([baseline.optimizer, baseline.loss], feed_dict)
            if episode % 50 == 0:
                saver = tf.train.Saver()
                save_path = saver.save(sess,
                                       f"/Users/Administrator/PycharmProjects/RDL_ex3/Ex_3/{env.spec.id}_Finetune/{env.spec.id}_Finetune.ckpt")
        with open('rewards_baseline.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(avg_reward_history)


if __name__ == '__main__':
    run()

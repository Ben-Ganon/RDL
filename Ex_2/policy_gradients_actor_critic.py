import random

import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
import csv
# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))
render = False
render_mode = "human" if render else None
env = gym.make('CartPole-v1', render_mode=render_mode)
Q_SIZE = 64
V_SIZE = 32

# value network that calculates the expected return for each state
class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.target = tf.placeholder(tf.float32, name="target")
            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            # Value function network with 3 layers of size V_SIZE, V_SIZE, 1
            self.W1 = tf.get_variable("W1", [self.state_size, V_SIZE], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [V_SIZE], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [V_SIZE, V_SIZE], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [V_SIZE], initializer=tf2_initializer)
            self.W3 = tf.get_variable("W3", [V_SIZE, 1], initializer=tf2_initializer)
            self.b3 = tf.get_variable("b3", [1], initializer=tf2_initializer)
            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            # self.output = tf.nn.relu(self.Z2)
            # do loss with mse loss
            self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.target))
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

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=1)
            # Policy network with 3 layers of size Q_SIZE, Q_SIZE, action_size
            self.W1 = tf.get_variable("W1", [self.state_size, Q_SIZE], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [Q_SIZE], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [Q_SIZE, Q_SIZE], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [Q_SIZE], initializer=tf2_initializer)
            self.W3 = tf.get_variable("W3", [Q_SIZE, action_size], initializer=tf2_initializer)
            self.b3 = tf.get_variable("b3", [action_size], initializer=tf2_initializer)
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


def run(seed):
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n
    np.random.seed(seed)
    tf.set_random_seed(seed)
    max_episodes = 1500
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0001
    learning_rate_baseline = learning_rate * 8


    # Initialize the policy network
    tf.reset_default_graph()
    policy_network = PolicyNetwork(state_size, action_size, learning_rate)
    value_network = ValueNetwork(state_size, learning_rate_baseline)

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        average_reward_history = []

        for episode in range(max_episodes):
            state = env.reset()[0]
            state = state.reshape([1, state_size])
            iteration_discount = 1.0
            for step in range(max_steps):
                actions_distribution = sess.run(policy_network.actions_distribution, {policy_network.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])
                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                if render:
                    env.render()
                reward = reward# convert to float
                value = sess.run(value_network.output, {value_network.state: state})
                next_value = discount_factor * sess.run(value_network.output, {value_network.state: next_state}) if not done else 0
                td_target = reward + next_value
                td_error = td_target - value
                # td_target = td_target * iteration_discount
                # td_error = td_error * iteration_discount


                # do w <- w + alpha * I * delta * grad v_tag(s)
                # do theta <- theta + alpha * I * delta * grad log π(a|s)
                _, loss = sess.run([value_network.optimizer, value_network.loss], {value_network.state: state, value_network.target: td_target})
                _, loss = sess.run([policy_network.optimizer, policy_network.loss], {policy_network.state: state, policy_network.R_t: td_error, policy_network.action: action_one_hot})

                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                        average_reward_history.append(average_rewards)
                    if episode % 20 == 0:
                        print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                        with open('rewards_actor_critic.csv', 'w') as f:
                            csvwriter = csv.writer(f)
                            csvwriter.writerow(average_reward_history)
                        f.close()
                        return "Done!"
                    break
                iteration_discount = iteration_discount * discount_factor
                state = next_state

            if solved:
                break
    return "not done!"
            #
            # Compute Rt for each time-step t and update the network's weights
            # for t, transition in enumerate(episode_transitions):
            #     total_discounted_return = sum(discount_factor ** i * (t.reward - t.value) for i, t in enumerate(episode_transitions[t:])) # Rt
            #     feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
            #     _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)


if __name__ == '__main__':
    seeds = [24, 25, 26, 27, 28, 42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for seed in random.choices(seeds, k=len(seeds)):
        print("seed: ", seed)
        value = run(seed)
        if value == "Done!":
            break
    # run(11)



# self.W1 = tf.get_variable("W1", [self.state_size, V_SIZE], initializer=tf2_initializer)
#             self.b1 = tf.get_variable("b1", [V_SIZE], initializer=tf2_initializer)
#             self.W2 = tf.get_variable("W2", [V_SIZE, 1], initializer=tf2_initializer)
#             self.b2 = tf.get_variable("b2", [1], initializer=tf2_initializer)


#
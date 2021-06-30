# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import tensorflow as tf
import numpy as np
import random
import time
import queue
import pickle
import matplotlib.pyplot as plt
import math
from PIL import Image

from train.rl_agents.td3.single_task.without_attention.rl_config import hyperParameters

# import error on dell pc
# from .rl_config import hyperParameters


# ==============================================================================
# ----------------------------- Functions --------------------------------------
# ==============================================================================
def get_split_batch(batch):
    """
    memory.sample() returns a batch of experiences, but we want an array
    for each element in the memory (s, a, r, s', done)
    """
    states_mb = np.array([each[0][0] for each in batch])
    # print(states_mb.shape) #shape 64*84*84*1 after reshaping im_final -- 64 is the batch size
    actions_mb = np.array([each[0][1] for each in batch])
    # print(actions_mb.shape)
    rewards_mb = np.array([each[0][2] for each in batch])
    # print(rewards_mb.shape) #shape (64,)
    next_states_mb = np.array([each[0][3] for each in batch])
    # print(next_states_mb.shape)
    dones_mb = np.array([each[0][4] for each in batch])
    return states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb


def OU(action, mu=0, theta=0.15, sigma=0.3):
    # noise = np.ones(action_dim) * mu
    noise = theta * (mu - action) + sigma * np.random.randn(1)
    # noise = noise + d_noise
    return noise


def map_action(action):
    """ [0, 1] -> [0, 9]"""
    target_speed = np.clip(action[0] - action[1], 0, 1)
    target_speed = target_speed * 9
    return target_speed


# ==============================================================================
# -- Classes -------------------------------------------------------------------
# ==============================================================================
# ==============================================================================
# -- Function approximation models ---------------------------------------------
# ==============================================================================
class ActorNetwork():
    def __init__(self, state_size, action_size, tau, learning_rate, net_name, rl_config=None):
        self.state_size = state_size
        self.action_size = action_size

        if rl_config:
            self.rl_config = rl_config
        else:  # default is training version
            self.rl_config = hyperParameters

        # use decayed learning rate
        self.global_step = tf.Variable(0, trainable=False)

        # get params from rl_config
        decay_steps = self.rl_config.decay_steps
        decay_rate = self.rl_config.decay_rate

        self.learning_rate = tf.train.exponential_decay(learning_rate,  # start learning rate
                                                        global_step=self.global_step,
                                                        decay_steps=decay_steps,
                                                        decay_rate=decay_rate,
                                                        staircase=True)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.tau = tau

        self.state_inputs, self.actor_variables, self.action = self.build_actor_network(net_name)
        self.state_inputs_target, self.actor_variables_target, self.action_target = self.build_actor_network(
            'actor_target')

        self.action_gradients = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name="action_gradients")
        self.actor_gradients = tf.compat.v1.gradients(self.action, self.actor_variables, -self.action_gradients)
        self.update_target_op = [self.actor_variables_target[i].assign(
            tf.multiply(self.actor_variables[i], self.tau) + tf.multiply(self.actor_variables_target[i], 1 - self.tau))
                                 for i in range(len(self.actor_variables))]

        # todo check global step
        self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.actor_variables),
                                                       global_step=self.global_step)

    def split_input(self, state):  # state:[batch, 31]
        rl_config = hyperParameters
        ego_state = tf.reshape(state[:, 0:rl_config.ego_feature_num], [-1, rl_config.ego_feature_num])
        npc_state = tf.reshape(state[:, rl_config.ego_feature_num:], [-1, rl_config.npc_num, rl_config.npc_feature_num])
        return ego_state, npc_state

    def build_actor_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state_inputs")
            ego_state, npc_state = self.split_input(state_inputs)
            # calculate action
            encoder_1 = tf.layers.dense(inputs=npc_state,
                                        units=64,
                                        activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="encoder_1")
            encoder_2 = tf.layers.dense(inputs=encoder_1,
                                        units=64,
                                        activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="encoder_2")
            concat = tf.concat([encoder_2[:, i] for i in range(5)], axis=1, name="concat")
            fc_1 = tf.concat([ego_state, concat], axis=1, name="fc_1")
            fc_2 = tf.layers.dense(inputs=fc_1,
                                   units=256,
                                   activation=tf.nn.tanh,
                                   kernel_initializer=tf.variance_scaling_initializer(),
                                   name="fc_2")
            # action output
            action_1 = tf.layers.dense(inputs=fc_2,
                                       units=256,
                                       activation=tf.nn.tanh,
                                       kernel_initializer=tf.variance_scaling_initializer(),
                                       name="action_1")
            action_2 = tf.layers.dense(inputs=action_1,
                                       units=256,
                                       activation=tf.nn.tanh,
                                       kernel_initializer=tf.variance_scaling_initializer(),
                                       name="action_2")
            speed_up = tf.layers.dense(inputs=action_2,
                                       units=1,
                                       activation=tf.nn.sigmoid,
                                       kernel_initializer=tf.variance_scaling_initializer(),
                                       name="speed_up")
            slow_down = tf.layers.dense(inputs=action_2,
                                        units=1,
                                        activation=tf.nn.sigmoid,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="slow_down")
            action = tf.concat([speed_up, slow_down], axis=1, name="action")

        actor_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return state_inputs, actor_variables, tf.squeeze(action)

    def get_action(self, sess, state):
        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))
        action = sess.run(self.action, feed_dict={
            self.state_inputs: state
        })
        return action

    def get_action_noise(self, sess, state, rate=1.):
        """
        Get an OU noise.

        :param sess:
        :param state:
        :param rate:
        :return:
        """
        rate = np.clip(rate, 0., 1.)
        # get original action
        action = self.get_action(sess, state)
        # print("original action: ", action)

        # todo fix this
        # apply OU noise
        speed_up_noised = action[0] + OU(action[0], mu=0.6, theta=0.15, sigma=0.3) * rate
        slow_down_noised = action[1] + OU(action[1], mu=0.2, theta=0.15, sigma=0.05) * rate
        action_noise = np.squeeze(
            np.array([np.clip(speed_up_noised, 0.01, 0.99), np.clip(slow_down_noised, 0.01, 0.99)]))
        print("noised action: ", action_noise)

        return action_noise

    def get_action_target(self, sess, state):
        target_noise = 0.01
        action_target = sess.run(self.action_target, feed_dict={
            self.state_inputs_target: state
        })
        action_target_smoothing = np.clip(action_target + np.random.rand(2) * target_noise, 0.01, 0.99)
        return action_target_smoothing

    def train(self, sess, state, action_gradients):

        # # debug global step and lr
        # print('global_step: ', sess.run(self.global_step))
        # print('lr: ', sess.run(self.learning_rate))

        sess.run(self.optimize, feed_dict={
            self.state_inputs: state,
            self.action_gradients: action_gradients
        })

        # # debug global step and lr
        # print('global_step: ', sess.run(self.global_step))
        # print('lr: ', sess.run(self.learning_rate))
        # print('')

    def update_target(self, sess):
        sess.run(self.update_target_op)


class CriticNetwork():
    def __init__(self, state_size, action_size, tau, learning_rate, net_name, rl_config=None):
        self.state_size = state_size
        self.action_size = action_size

        # use decayed learning rate
        if rl_config:
            self.rl_config = rl_config
        else:
            self.rl_config = hyperParameters

        # use decayed learning rate
        self.global_step = tf.Variable(0, trainable=False)

        # todo if lrc and lra using different decay rate
        # get params from rl_config
        decay_steps = self.rl_config.decay_steps
        decay_rate = self.rl_config.decay_rate

        self.learning_rate = tf.train.exponential_decay(learning_rate,  # start learning rate
                                                        global_step=self.global_step,
                                                        decay_steps=decay_steps,
                                                        decay_rate=decay_rate,
                                                        staircase=True)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.optimizer_2 = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.tau = tau

        self.state_inputs, self.action, self.critic_variables, self.q_value = self.build_critic_network(net_name)
        self.state_inputs_target, self.action_target, self.critic_variables_target, self.q_value_target = self.build_critic_network(
            net_name + "_target")

        self.target = tf.compat.v1.placeholder(tf.float32, [None])
        self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.absolute_errors = tf.abs(self.target - self.q_value)  # for updating sumtree
        self.loss = tf.reduce_mean(
            self.ISWeights * tf.compat.v1.losses.huber_loss(labels=self.target, predictions=self.q_value))
        self.loss_2 = tf.reduce_mean(tf.compat.v1.losses.huber_loss(labels=self.target, predictions=self.q_value))
        # self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.critic_variables])
        # self.loss = tf.reduce_mean(tf.square(self.target - self.q_value)) + 0.01*self.l2_loss + 0*self.ISWeights
        self.optimize = self.optimizer.minimize(self.loss)
        self.optimize_2 = self.optimizer_2.minimize(self.loss_2)
        self.update_target_op = [self.critic_variables_target[i].assign(
            tf.multiply(self.critic_variables[i], self.tau) + tf.multiply(self.critic_variables_target[i],
                                                                          1 - self.tau)) for i in
                                 range(len(self.critic_variables))]
        self.action_gradients = tf.gradients(self.q_value, self.action)

    def split_input(self, state):  # state:[batch, 31]
        rl_config = hyperParameters
        ego_state = tf.reshape(state[:, 0:rl_config.ego_feature_num], [-1, rl_config.ego_feature_num])
        npc_state = tf.reshape(state[:, rl_config.ego_feature_num:], [-1, rl_config.npc_num, rl_config.npc_feature_num])
        return ego_state, npc_state

    def build_critic_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state_inputs")
            action_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name="action_inputs")
            ego_state, npc_state = self.split_input(state_inputs)
            # calculate q-value
            encoder_1 = tf.layers.dense(inputs=npc_state,
                                        units=64,
                                        activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="encoder_1")
            encoder_2 = tf.layers.dense(inputs=encoder_1,
                                        units=64,
                                        activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="encoder_2")
            concat = tf.concat([encoder_2[:, i] for i in range(5)], axis=1, name="concat")
            fc_1 = tf.concat([ego_state, concat], axis=1, name="fc_1")
            fc_2 = tf.layers.dense(inputs=fc_1,
                                   units=256,
                                   activation=tf.nn.tanh,
                                   kernel_initializer=tf.variance_scaling_initializer(),
                                   name="fc_2")
            # state+action merge
            action_fc = tf.layers.dense(inputs=action_inputs,
                                        units=256, activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="action_fc")
            merge = tf.concat([fc_2, action_fc], axis=1, name="merge")
            merge_fc = tf.layers.dense(inputs=merge,
                                       units=256, activation=tf.nn.tanh,
                                       kernel_initializer=tf.variance_scaling_initializer(),
                                       name="merge_fc")
            # q value output
            q_value = tf.layers.dense(inputs=merge_fc,
                                      units=1, activation=None,
                                      kernel_initializer=tf.variance_scaling_initializer(),
                                      name="q_value")
        critic_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return state_inputs, action_inputs, critic_variables, tf.squeeze(q_value)

    def get_q_value_target(self, sess, state, action):
        return sess.run(self.q_value_target, feed_dict={
            self.state_inputs_target: state,
            self.action_target: action
        })

    def get_gradients(self, sess, state, action):
        return sess.run(self.action_gradients, feed_dict={
            self.state_inputs: state,
            self.action: action
        })

    def train(self, sess, state, action, target, ISWeights):
        _, _, loss, absolute_errors = sess.run([self.optimize, self.optimize_2, self.loss, self.absolute_errors],
                                               feed_dict={
                                                   self.state_inputs: state,
                                                   self.action: action,
                                                   self.target: target,
                                                   self.ISWeights: ISWeights
                                               })
        return loss, absolute_errors

    def update_target(self, sess):
        sess.run(self.update_target_op)


# ==============================================================================
# -- Prioritized Replay --------------------------------------------------------
# ==============================================================================
class SumTree(object):
    """
        This SumTree code is modified version of Morvan Zhou:
        https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py"""

    data_pointer = 0

    # initialise tree with all nodes = 0 and data with all values =0

    def __init__(self, capacity):

        self.capacity = capacity
        # number of leaf nodes that contains experiences
        # generate the tree with all nodes values =0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity

        self.tree = np.zeros(
            2 * capacity - 1)  # was initally np.zeroes, but after making memory_size>pretain_length, it had to be adjusted

        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            # overwrite
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # change = new priority - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through the tree // update whole tree
        while tree_index != 0:
            """
                        Here we want to access the line above
                        THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
                            0
                           / \
                          1   2
                         / \ / \
                        3  4 5  [6]
                        If we are in leaf at index 6, we updated the priority score
                        We need then to update index 2 node
                        So tree_index = (tree_index - 1) // 2
                        tree_index = (6-1)//2
                        tree_index = 2 (because // round the result)
                        """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        # here we get the leaf_index, priority value of that leaf and experience associated with that index
        """
                        Tree structure and array storage:
                        Tree index:
                             0         -> storing priority sum
                            / \
                          1     2
                         / \   / \
                        3   4 5   6    -> storing priority for experiences
                        Array type for storing:
                        [0,1,2,3,4,5,6]
                        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        # data_index is the child index that we want ro get the data from, leaf index is it's parent ndex
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class Memory(object):
    """
    This SumTree code is modified version of:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    def __init__(self, capacity, pretrain_length, action_size):
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """

        self.tree = SumTree(capacity)
        self.pretrain_length = pretrain_length
        self.action_size = action_size
        self.possible_actions = np.identity(self.action_size, dtype=int).tolist()
        # hyperparamters
        self.absolute_error_upper = 1.  # clipped abs error
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
        self.PER_b_increment_per_sampling = 0.001

    def store(self, experience):
        """
         Store a new experience in the tree with max_priority
         When training the priority is to be ajusted according with the prediction error
        """
        # find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        # use minimum priority =1
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        # create sample array to contain minibatch
        memory_b = []
        if n > self.tree.capacity:
            print("Sample number more than capacity")
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        # calc the priority segment, divide Range into n ranges
        priority_segment = self.tree.total_priority / n

        # increase PER_b each time we sample a minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        # calc max_Weights
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        # print(np.min(self.tree.tree[-self.tree.capacity:]))
        # print(self.tree.total_priority)
        # print("pmin =" , p_min)
        max_weight = (p_min * n) ** (-self.PER_b)
        # print("max weight =" ,max_weight)

        for i in range(n):
            # A value is uniformly sampled from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)
            # print("priority =", priority)

            sampling_probabilities = priority / self.tree.total_priority
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            # print("weights =", b_ISWeights[i,0])
            # print(b_ISWeights.shape) shape(64,1)

            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):

        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def fill_memory(self, env):
        '''Fill memory with experiences. If autopilot == True experience is given by autopilot,
        otherwise agent takes random actions'''
        print("Started to fill memory")
        state = env.reset()
        for i in range(self.pretrain_length):
            if i % 500 == 0:
                print(i, "experiences stored")
            action = np.array([random.random(), random.random()])
            # target_speed = map_action(action)
            # next_state, reward, done, _ = env.step(target_speed)

            next_state, reward, done, _ = env.step(action)

            experience = state, action, reward, next_state, done
            # print(action)
            self.store(experience)

            if done:
                state = env.reset()
            else:
                state = next_state

        print('Finished filing memory. %s experiences stored.' % self.pretrain_length)

    def save_memory(self, filename, object):
        handle = open(filename, "wb")
        pickle.dump(object, handle)

    def load_memory(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

import os
import numpy as np
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


class DebugConfig:
    """
    This is parameters for experiment debug.
    """
    # Training parameters
    total_episodes = 100
    noised_episodes = 20
    max_steps = 300
    batch_size = 1024  # 256
    train_frequency = 500  # 2

    # NN architecture
    ego_feature_num = 4
    npc_num = 5
    npc_feature_num = 4
    state_size = ego_feature_num + npc_num * npc_feature_num
    action_size = 2
    lra = 2e-5
    lrc = 1e-4

    # Fixed Q target hyper parameters
    tau = 1e-3

    # exploration hyperparamters for ep. greedy. startegy
    explore_start = 0.75  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    explore_step = 40000  # 40k steps
    decay_rate = (explore_start - explore_stop) / explore_step  # exponential decay rate for exploration prob

    # Q LEARNING hyperparameters
    gamma = 0.99  # Discounting rate
    pretrain_length = 500  # Number of experiences stored in the Memory when initialized for the first time --INTIALLY 100k
    memory_size = 200000  # Number of experiences the Memory can keep  --INTIALLY 100k
    load_memory = False  # If True load memory, otherwise fill the memory with new data

    # ==================================================
    # output paths
    tag = 'debug'
    output_path = os.path.join('./outputs', tag, TIMESTAMP)

    memory_path = os.path.join(output_path, 'rl_replay_memory')
    # os.makedirs(memory_path, exist_ok=True)

    memory_load_path = os.path.join(memory_path, 'memory.pkl')
    memory_save_path = os.path.join(memory_path, 'memory.pkl')

    # model saving
    model_save_frequency = 2  # frequency to save the model. 0 means not to save
    model_save_frequency_no_paste = 500  # ???

    # frequency to check best models
    model_save_frequency_high_success = 10

    model_test_frequency = 10
    model_test_eps = 10  # ???

    # final model save path
    model_save_path = os.path.join(output_path, 'final_model', 'final_model.ckpt')
    # checkpoint save path
    model_ckpt_path = os.path.join(output_path, 'checkpoints')
    # best model
    best_model_path = os.path.join(output_path, 'best_models')


class hyperParameters:
    """
    Hyperparameters for RL agent
    """

    # Training parameters
    total_episodes = 10000
    noised_episodes = 2000
    # todo add setter for env
    max_steps = 300
    batch_size = 1024  # 256, 512, 1024
    train_frequency = 2  # 2

    # td3
    policy_delayed = 2

    # NN architecture
    ego_feature_num = 4  # 9 for abs_all, 4 for sumo and sumo_1
    npc_num = 5
    npc_feature_num = 6
    state_size = ego_feature_num + npc_num * npc_feature_num

    action_size = 2

    # Fixed Q target hyper parameters
    tau = 1e-3

    # exploration hyper-parameters for epsilon-greedy strategy
    explore_start = 0.5  # exploration probability at start
    explore_stop = 0.05  # minimum exploration probability
    explore_step = 20000  # 40k, 40000
    # decay_rate = (explore_start - explore_stop) / explore_step  # exponential decay rate for exploration prob

    # Q LEARNING hyperparameters
    gamma = 0.99  # Discounting rate
    pretrain_length = 10000  # Number of experiences stored in the Memory when initialized for the first time --INTIALLY 100k
    memory_size = 500000  # Number of experiences the Memory can keep  --INTIALLY 100k
    load_memory = False  # If True load memory, otherwise fill the memory with new data

    # ==================================================
    # output paths
    # tag = 'CarlaEnv3'
    tag = 'CarlaEnv4'

    # model saving
    model_save_frequency = 50  # frequency to save the model. 0 means not to save
    model_save_frequency_no_paste = 500  # ???

    # frequency to check best models
    model_save_frequency_high_success = 20

    model_test_frequency = 10
    model_test_eps = 10  # ???

    # ================   Decay learning rate   ================

    lra = 2e-5  # 2e-5
    lrc = 5e-5  # 1e-4

    # todo this number is determined by downsample factor
    guessing_episode_length = 200
    # decay after certain number of episodes
    decay_episodes = 1500
    decay_steps = guessing_episode_length / train_frequency * decay_episodes
    decay_rate = 1 / 2.15  # 2.15 = 10^(1/3)

    def __init__(self, args=None):

        # todo need to fix api of the rl_utils
        self.state_size = self.ego_feature_num + self.npc_num * self.npc_feature_num

        self.generate_output_path(args)

    def generate_output_path(self, args=None):

        if args:
            if args.tag:
                output_path = os.path.join('./outputs', args.route_option, self.tag, args.tag, TIMESTAMP)
            else:
                output_path = os.path.join('./outputs', args.route_option, self.tag, TIMESTAMP)
        else:
            output_path = os.path.join('./outputs/please_check', self.tag, TIMESTAMP)

        self.output_path = output_path

        self.memory_path = os.path.join(output_path, 'rl_replay_memory')
        # os.makedirs(memory_path, exist_ok=True)

        self.memory_load_path = os.path.join(self.memory_path, 'memory.pkl')
        self.memory_save_path = os.path.join(self.memory_path, 'memory.pkl')

        # checkpoint save path
        self.model_ckpt_path = os.path.join(output_path, 'checkpoints')

        # best model
        self.best_model_path = os.path.join(output_path, 'best_models')

        # final model save path
        self.model_save_path = os.path.join(output_path, 'final_model', 'final_model.ckpt')


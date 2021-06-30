"""
Add some new function into rl_agent:

 - add args to set training options
"""

import glob
import os
import sys

# ================   Append Project Path   ================
path = os.getcwd()
# index = path.split('/').index('gym-carla') if 'gym-carla' in path.split('/') else -1

source_path = 'train'
index = path.split('/').index(source_path) if source_path in path.split('/') else -1

proj_path = '/' + os.path.join(*path.split('/')[:index+1]) + '/'
sys.path.append(proj_path)

# ================   Append CARLA Path   ================
from train.gym_carla.config.carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

# import glob
# import os
# import sys

carla_root = os.path.join(root_path, 'CARLA_' + carla_version)
carla_path = os.path.join(carla_root, 'PythonAPI')
sys.path.append(carla_path)
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla'))
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla/agents'))

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import numpy as np
import tensorflow as tf

import argparse
import logging
import pickle

from rl_utils import ActorNetwork, CriticNetwork, Memory, get_split_batch
from rl_config import DebugConfig, hyperParameters

from train.gym_carla.envs.carla_env4_fixed import CarlaEnv4Fixed
from train.rl_agents.td3.single_task.without_attention.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def init_tensorflow():
    # tensorflow init
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto


def train_loop(args, rl_config):
    """"""

    configProto = init_tensorflow()

    # fixed version
    env = CarlaEnv4Fixed(
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        route_option=args.route_option,
        state_option=args.state_option,
        tm_seed=args.tm_seed,
        initial_speed=args.initial_speed,
        use_tls_control=args.use_tls_control,
        switch_traffic_flow=args.switch_traffic_flow,
        multi_task=args.multi_task,
        attention=False,
        debug=False,
        # training=True,
        tf_randomize=True,  # True for training, False to run with fixed traffic flow
        collision_prob_decay=True,  # if True, enable collision prob decay
        tf_params_decay=True,  # traffic params decay, shrink to a compact range with training proceeding
    )

    # init the OU action noise
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.array([0., 0.]),
        sigma=np.array([0.3, 0.3]),
        theta=np.array([0.15, 0.15]),
        dt=env.simulator_timestep_length,  # set to env timestep, default is 0.05
        x0=np.array([0., 0.]),
    )

    actor = ActorNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lra, 'actor')
    critic_1 = CriticNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lrc, 'critic_1')
    critic_2 = CriticNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lrc, 'critic_2')

    # tensorflow summary for tensorboard visualization
    log_path = os.path.join(rl_config.output_path, 'log')
    writer = tf.compat.v1.summary.FileWriter(log_path)
    # losses
    # tf.compat.v1.summary.scalar("Loss", critic.loss)
    # tf.compat.v1.summary.histogram("ISWeights", critic.ISWeights)
    # write_op = tf.compat.v1.summary.merge_all()
    saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    # initialize memory and fill it with examples, for prioritized replay
    os.makedirs(rl_config.memory_path, exist_ok=True)

    memory = Memory(rl_config.memory_size, rl_config.pretrain_length, rl_config.action_size)
    if rl_config.load_memory:
        memory = memory.load_memory(rl_config.memory_load_path)
        print("MEMORY: Memory Loaded")
    else:
        memory.fill_memory(env)
        memory.save_memory(rl_config.memory_save_path, memory)
        print("MEMORY: Memory Filled")

    # make result folder
    result_path = os.path.join(rl_config.output_path, 'rl_results')
    os.makedirs(result_path, exist_ok=True)

    # Reinforcement Learning loop
    with tf.compat.v1.Session(config=configProto) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # update param of target network
        writer.add_graph(sess.graph)
        actor.update_target(sess)
        critic_1.update_target(sess)
        critic_2.update_target(sess)

        recent_rewards = []  # rewards from recent 100 episodes
        avarage_rewards = []  # average reward of recent 100 episodes
        recent_success = []
        recent_success_rate = []

        # reset the episode number
        env.reset_episode_count()

        for episode in range(1, rl_config.total_episodes + 1):
            # move the vehicle to a spawn_point and return state
            state = env.reset()
            episode_reward = 0
            done = False

            # epsilon refers to the decay rate of the noise
            min_factor = 0.25
            noised_episodes = rl_config.noised_episodes
            if episode > noised_episodes:
                epsilon = 0.
            else:
                epsilon = (min_factor - 1) / noised_episodes * episode + 1

            # rollout single episode
            while True:

                # old version of OU-noised action
                # action_noise = actor.get_action_noise(sess, state, rate=epsilon)

                # original action
                action = actor.get_action(sess, state)
                action = np.squeeze(action)

                # add noise on original action
                action_noise = action + ou_noise() * epsilon

                action_new = action_noise
                next_state, reward, done, info = env.step(action_new)

                episode_reward += reward
                experience = state, action_noise, reward, next_state, done
                memory.store(experience)

                # Lets learn
                if env.episode_step_number % rl_config.train_frequency == 0:
                    # "Delayed" Policy Updates
                    policy_delayed = rl_config.policy_delayed
                    for _ in range(policy_delayed):
                        # First we need a mini-batch with experiences (s, a, r, s', done)
                        tree_idx, batch, ISWeights_mb = memory.sample(rl_config.batch_size)
                        # print(ISWeights_mb)
                        s_mb, a_mb, r_mb, next_s_mb, dones_mb = get_split_batch(batch)
                        # print(a_mb.shape)

                        # Get q_target values for next_state from the critic_target
                        a_target_next_state = actor.get_action_target(sess, next_s_mb)  # with Target Policy Smoothing
                        q_target_next_state_1 = critic_1.get_q_value_target(sess, next_s_mb, a_target_next_state)
                        q_target_next_state_2 = critic_2.get_q_value_target(sess, next_s_mb, a_target_next_state)
                        q_target_next_state = np.minimum(q_target_next_state_1, q_target_next_state_2)

                        # Set Q_target = r if the episode ends at s+1, otherwise Q_target = r + gamma * Qtarget(s',a')
                        target_Qs_batch = []
                        for i in range(0, len(dones_mb)):
                            terminal = dones_mb[i]
                            # if we are in a terminal state. only equals reward
                            if terminal:
                                target_Qs_batch.append((r_mb[i]))
                            else:
                                # take the Q taregt for action a'
                                target = r_mb[i] + rl_config.gamma * q_target_next_state[i]
                                target_Qs_batch.append(target)
                        targets_mb = np.array([each for each in target_Qs_batch])

                        # critic train
                        if len(a_mb.shape) > 2:
                            a_mb = np.squeeze(a_mb, axis=1)
                        loss, absolute_errors = critic_1.train(sess, s_mb, a_mb, targets_mb, ISWeights_mb)
                        loss_2, absolute_errors_2 = critic_2.train(sess, s_mb, a_mb, targets_mb, ISWeights_mb)
                        # print('loss:',loss)

                    # actor train
                    a_for_grad = actor.get_action(sess, s_mb)
                    a_gradients = critic_1.get_gradients(sess, s_mb, a_for_grad)
                    # print(a_gradients)
                    actor.train(sess, s_mb, a_gradients[0])
                    # target train
                    actor.update_target(sess)
                    critic_1.update_target(sess)
                    critic_2.update_target(sess)

                    # update replay memory priorities
                    memory.batch_update(tree_idx, absolute_errors)

                # store episode data or continue
                if done:
                    # get and save traffic flow distribution
                    dist_dict = env.traffic_flow.tf_distribution
                    with open(os.path.join(result_path, "tf_distribution" + ".pkl"), "wb") as f:
                        pickle.dump(dist_dict, f, pickle.HIGHEST_PROTOCOL)

                    # save checkpoint model
                    if (episode) % rl_config.model_save_frequency == 0:
                        ckpt_path = os.path.join(rl_config.model_ckpt_path, 'checkpoint_' + str(episode) + '.ckpt')
                        saver.save(sess, ckpt_path)
                        print('model saved')

                    # visualize reward data
                    recent_rewards.append(episode_reward)
                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)
                    avarage_rewards.append(np.mean(recent_rewards))
                    avarage_rewards_data = np.array(avarage_rewards)
                    d = {"avarage_rewards": avarage_rewards_data}

                    with open(os.path.join(result_path, "reward_data" + ".pkl"), "wb") as f:
                        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

                    # visualize success rate data
                    if info['exp_state'] == 'success':
                        recent_success.append(1)
                    else:
                        recent_success.append(0)
                    if len(recent_success) > 100:
                        recent_success.pop(0)

                    # todo add a method to better save best models
                    # calculate success rate and save best models
                    avarage_success_rate = recent_success.count(1) / len(recent_success)

                    # ===============   save best model   ===============
                    # todo add this to rl_config
                    if episode > 4000:
                        if episode % rl_config.model_save_frequency_high_success == 0:
                            if rl_config.model_save_frequency and avarage_success_rate >= 0.98:  # ref success rate
                                best_model_path = os.path.join(rl_config.best_model_path,
                                                               'best_model_' + str(episode) + ".ckpt")
                                saver.save(sess, best_model_path)
                                print('best model saved')

                    recent_success_rate.append(avarage_success_rate)
                    recent_success_rate_data = np.array(recent_success_rate)
                    d = {"recent_success_rates": recent_success_rate_data}
                    with open(os.path.join(result_path, "success_rate_data" + ".pkl"), "wb") as f:
                        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
                    # print results on terminal
                    print('---' * 15)
                    print("avarage_rewards:", np.mean(recent_rewards))
                    print("recent_success_rate:", avarage_success_rate)
                    print(episode, 'episode finished. Episode total reward:', episode_reward)
                    print(episode, 'episode finished. Episode total steps:', env.episode_step_number, 'Episode time:',
                          env.episode_time)
                    print('---' * 15)
                    break
                else:
                    state = next_state


def test_loop(args, rl_config, test_num=100):
    """
    If you wish to run evaluation in training env, please set model_path to your trained dict.
    :param args:
    :param rl_config:
    :return:
    """

    model_path = ''

    configProto = init_tensorflow()

    actor = ActorNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lra, 'actor')
    # critic = CriticNetwork(rl_config.state_size, rl_config.action_size, rl_config.tau, rl_config.lrc, 'critic')
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(config=configProto) as sess:
        # load network
        # saver = tf.compat.v1.train.import_meta_graph(rl_config.model_save_path + '.meta')
        # saver.restore(sess, rl_config.model_save_path)

        # saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        if saver is None:
            print("did not load")

        # fixed version
        env = CarlaEnv4Fixed(
            carla_port=args.carla_port,
            tm_port=args.tm_port,
            route_option=args.route_option,
            state_option=args.state_option,
            tm_seed=args.tm_seed,
            initial_speed=args.initial_speed,
            use_tls_control=args.use_tls_control,
            switch_traffic_flow=args.switch_traffic_flow,
            multi_task=args.multi_task,
            attention=False,
            debug=False,
            # training=True,
            tf_randomize=True,  # True for training, False to run with fixed traffic flow
            collision_prob_decay=True,  # if True, enable collision prob decay
            tf_params_decay=True,  # traffic params decay, shrink to a compact range with training proceeding
        )

        # test_num = 100  # default test episode number
        test_ep = 0
        step = 0  # step in one episode
        success = 0
        failure = 0
        collision = 0
        time_exceed = 0
        episode_time_record = []

        episode_reward = 0
        state = env.reset()
        done = False
        while test_ep < test_num:
            action = actor.get_action(sess, state)
            # target_speed = map_action(action)
            # state, reward, done, info = env.step(target_speed)

            # if use_action_modifier:
            #     c = env.query_constraint_value(state)
            #     action_new = action_modifier(state, action, c)
            # else:
            #     # not use action modifier
            #     action_new = action

            action_new = action
            state, reward, done, info = env.step(action_new)

            episode_reward += reward
            step += 1

            if done:
                # record result
                if info['exp_state'] == 'collision':
                    collision += 1
                    failure += 1
                elif info['exp_state'] == 'time_exceed':
                    time_exceed += 1
                    failure += 1
                else:
                    # get episode time
                    episode_time_record.append(env.episode_time)
                    success += 1
                # print
                print(test_ep, "EPISODE ended", "TOTAL REWARD {:.4f}".format(episode_reward), 'Result:',
                      info['exp_state'])
                print('total step of this episode: ', step)
                state = env.reset()  # init state
                episode_reward = 0
                step = 0
                done = False
                test_ep += 1

        print('-*' * 15, ' result ', '-*' * 15)
        print('success: ', success, '/', test_num)
        print('collision: ', collision, '/', test_num)
        print('time_exceed: ', time_exceed, '/', test_num)
        print('average time: ', np.mean(episode_time_record))


def render_loop(args):
    if args.debug:
        rl_config = DebugConfig
    else:
        rl_config = hyperParameters

    # todo get state dimension according to state option
    # if args.state_option not in state_options.keys():
    #     raise ValueError('Wrong state option, please check')
    # elif args.state_option == 'sumo':
    #     rl_config.ego_feature_num = int(4)
    # elif args.state_option == 'kinetics':
    #     rl_config.ego_feature_num = int(3)

    # if args.state_option == 'absolute_all':

    # rl_config.tag = rl_config.tag + '_state_' + args.state_option

    # new 2021.04.13, append route option into output path
    rl_config.tag = 'state_' + args.state_option

    try:
        #
        # training or testing
        if args.test:
            print('---------Begin TESTing---------')
            test_loop(args, rl_config)
        else:
            print('---------Begin TRAINing---------')
            train_loop(args, rl_config(args))
    finally:
        pass


def main():
    """"""

    argparser = argparse.ArgumentParser(
        description='CARLA Intersection Scenario')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='test',
        help='test a trained model')
    argparser.add_argument(
        '--port',
        default=int(2000),
        type=int,
        dest='carla_port',
        help='carla simulator port number')
    argparser.add_argument(
        '--tm-port',
        default=int(8100),
        type=int,
        dest='tm_port',
        help='traffic manager port number')
    argparser.add_argument(
        '--route',
        default="left",
        dest='route_option',
        help='route option for single task')
    argparser.add_argument(
        '--state',
        default="sumo_1",  # ["sumo", 'absolute_all', "sumo_1"],
        dest='state_option',
        help='state representation option')
    argparser.add_argument(
        '--init-speed',
        default=None,
        type=float,
        dest='initial_speed',
        help='initial speed of ego vehicle')
    argparser.add_argument(
        '--seed',
        default=int(0),
        type=int,
        dest='tm_seed',
        help='traffic manager seed number')
    argparser.add_argument(
        '--tag',
        default=None,
        dest='tag',
        help='additional tag for current result path')
    argparser.add_argument(
        '--switch-tf',
        action='store_true',
        dest='switch_traffic_flow',
        help='whether switch traffic flow in single task training')
    argparser.add_argument(
        '--tls-control',
        action='store_true',
        dest='use_tls_control',
        help='whether use traffic lights control')
    argparser.add_argument(
        '--multi-task',
        action="store_true",
        dest='multi_task',
        help='whether run multi task training')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # print(__doc__)

    # add model path for test phase

    try:
        render_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

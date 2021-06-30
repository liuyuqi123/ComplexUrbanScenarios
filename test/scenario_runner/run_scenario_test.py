"""
A modified version of scenario_runner for scenario test.

"""

import glob
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import importlib
import inspect
import os
import signal
import sys
import time
import json

# =======================================================
# ================   Append CARLA path   ================
# from srunner.drl_code.carla_config import version_config
from train.gym_carla.config.carla_config import version_config

carla_version = version_config['carla_version']
root_path = version_config['root_path']

carla_root = os.path.join(root_path, 'CARLA_' + carla_version)
carla_path = os.path.join(carla_root, 'PythonAPI')
sys.path.append(carla_path)
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla/'))
sys.path.append(os.path.join(carla_root, 'PythonAPI/carla/agents'))

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
# =======================================================

import carla

# =======================================================
# append scenario_runner path
# this script is supposed to be put in parent folder
scenario_runner_path = os.getcwd()
sys.path.append(scenario_runner_path)

# add os variable, for getting *.xml file
os.environ['SCENARIO_RUNNER_ROOT'] = os.path.join(scenario_runner_path)

# this line to test whether scenario_runner path is successfully added
# list_of_config_files = glob.glob("{}/srunner/examples/*.xml".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))

# -------------------------------------------------------
# append gym_carla
gym_carla_path = os.path.abspath(os.path.join(scenario_runner_path, '..', '..'))  # + '/'
gym_carla_path = os.path.join(gym_carla_path, 'train')
sys.path.append(gym_carla_path)
sys.path.append(gym_carla_path + '/')
os.environ['GYM_CARLA_ROOT'] = gym_carla_path

import random
import xlwt
import pickle
import tensorflow as tf
import numpy as np
# =======================================================

from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser

# ==============   Import fixed modules   ==============
from srunner.drl_code.scenario_utils.scenario_route_manager import ScenarioRouteManager
from srunner.drl_code.scenario_utils.scenario_manager_fixed import ScenarioManagerFixed
from srunner.drl_code.scenario_utils.util_visualization import draw_waypoint
from srunner.drl_code.scenario_utils.carla_color import *

# Version of scenario_runner
VERSION = '0.9.10'

TIMESTAMP = "{0:%Y-%m-%d-Time%H-%M-%S}".format(datetime.now())


class ScenarioRunner(object):
    """
    This is the core scenario runner module. It is responsible for
    running (and repeating) a single scenario or a list of scenarios.

    Usage:
    scenario_runner = ScenarioRunner(args)
    scenario_runner.run()
    del scenario_runner
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0  # in Hz

    # CARLA world and scenario handlers
    world = None
    manager = None

    additional_scenario_module = None

    agent_instance = None
    module_agent = None

    def __init__(self, args):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self._args = args

        if args.timeout:
            self.client_timeout = float(args.timeout)

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(self._args.trafficManagerPort))

        # dist = pkg_resources.get_distribution("carla")
        # if LooseVersion(dist.version) < LooseVersion('0.9.10'):
        #     raise ImportError("CARLA version 0.9.10 or newer required. CARLA version found: {}".format(dist))

        # Load agent if requested via command line args
        # If something goes wrong an exception will be thrown by importlib (ok here)
        if self._args.agent is not None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        # self.manager = ScenarioManager(self._args.debug, self._args.sync, self._args.timeout)

        # ======== modified version of ScenarioManager
        self.manager = ScenarioManagerFixed(
            self._args.debug,
            self._args.sync,
            self._args.timeout,
            # ========  additional args  ========
            sync_tick=True,  # todo add this to the argparse
        )

        # Create signal handler for SIGINT
        self._shutdown_requested = False
        if sys.platform != 'win32':
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._start_wall_time = datetime.now()

        # tag output path
        self.tag = (self._args.tag + '/' + TIMESTAMP) if self._args.tag else TIMESTAMP

    def destroy(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world
        if self.client is not None:
            del self.client

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._shutdown_requested = True
        if self.manager:
            self.manager.stop_scenario()
            self._cleanup()
            if not self.manager.get_running_status():
                raise RuntimeError("Timeout occured during scenario execution")

    def _get_scenario_class_or_fail(self, scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """

        # Path of all scenario at "srunner/scenarios" folder + the path of the additional scenario argument
        scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        scenarios_list.append(self._args.additionalScenario)

        for scenario_file in scenarios_list:

            # Get their module
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            scenario_module = importlib.import_module(module_name)

            # And their members of type class
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                if scenario in member:
                    return member[1]

            # Remove unused Python paths
            sys.path.pop(0)

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        # Simulation still running and in synchronous mode?
        if self.world is not None and self._args.sync:
            try:
                # Reset to asynchronous mode
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except RuntimeError:
                sys.exit(-1)

        self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if not self._args.waitForEgo:
                    print("Destroying ego vehicle {}".format(self.ego_vehicles[i].id))
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles):
        """
        Spawn or update the ego vehicles
        """

        if not self._args.waitForEgo:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             actor_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)
                CarlaDataProvider.register_actor(self.ego_vehicles[i])

        # sync state
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _analyze_scenario(self, config):
        """
        Provide feedback about success/failure of a scenario
        """

        # Create the filename
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        json_filename = None
        config_name = config.name
        if self._args.outputDir != '':
            config_name = os.path.join(self._args.outputDir, config_name)

        if self._args.junit:
            junit_filename = config_name + current_time + ".xml"
        if self._args.json:
            json_filename = config_name + current_time + ".json"
        filename = None
        if self._args.file:
            filename = config_name + current_time + ".txt"

        if not self.manager.analyze_scenario(self._args.output, filename, junit_filename, json_filename):
            print("All scenario tests were passed successfully!")
        else:
            print("Not all scenario tests were successful")
            if not (self._args.output or filename or junit_filename):
                print("Please run with --output for further information")

    def _record_criteria(self, criteria, name):
        """
        Filter the JSON serializable attributes of the criterias and
        dumps them into a file. This will be used by the metrics manager,
        in case the user wants specific information about the criterias.
        """
        file_name = name[:-4] + ".json"

        # Filter the attributes that aren't JSON serializable
        with open('temp.json', 'w') as fp:

            criteria_dict = {}
            for criterion in criteria:

                criterion_dict = criterion.__dict__
                criteria_dict[criterion.name] = {}

                for key in criterion_dict:
                    if key != "name":
                        try:
                            key_dict = {key: criterion_dict[key]}
                            json.dump(key_dict, fp, sort_keys=False, indent=4)
                            criteria_dict[criterion.name].update(key_dict)
                        except TypeError:
                            pass

        os.remove('temp.json')

        # Save the criteria dictionary into a .json file
        with open(file_name, 'w') as fp:
            json.dump(criteria_dict, fp, sort_keys=False, indent=4)

    def _load_and_wait_for_world(self, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        if self._args.reloadWorld:
            self.world = self.client.load_world(town)
        else:
            # if the world should not be reloaded, wait at least until all ego vehicles are ready
            ego_vehicle_found = False
            if self._args.waitForEgo:
                while not ego_vehicle_found and not self._shutdown_requested:
                    vehicles = self.client.get_world().get_actors().filter('vehicle.*')
                    for ego_vehicle in ego_vehicles:
                        ego_vehicle_found = False
                        for vehicle in vehicles:
                            if vehicle.attributes['role_name'] == ego_vehicle.rolename:
                                ego_vehicle_found = True
                                break
                        if not ego_vehicle_found:
                            print("Not all ego vehicles ready. Waiting ... ")
                            time.sleep(1)
                            break

        self.world = self.client.get_world()

        if self._args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            self.world.apply_settings(settings)

            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(int(self._args.trafficManagerSeed))

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(self._args.trafficManagerPort))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()
        if CarlaDataProvider.get_map().name != town and CarlaDataProvider.get_map().name != "OpenDriveMap":
            print("The CARLA server uses the wrong map: {}".format(CarlaDataProvider.get_map().name))
            print("This scenario requires to use map: {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, config):
        """
        Load and run the scenario given by config
        """
        result = False
        if not self._load_and_wait_for_world(config.town, config.ego_vehicles):
            self._cleanup()
            return False

        if self._args.agent:
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            try:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig)
                config.agent = self.agent_instance
            except Exception as e:  # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return False

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            self._prepare_ego_vehicles(config.ego_vehicles)
            if self._args.openscenario:
                scenario = OpenScenario(world=self.world,
                                        ego_vehicles=self.ego_vehicles,
                                        config=config,
                                        config_file=self._args.openscenario,
                                        timeout=100000)
            elif self._args.route:
                scenario = RouteScenario(world=self.world,
                                         config=config,
                                         debug_mode=self._args.debug)
            else:
                scenario_class = self._get_scenario_class_or_fail(config.type)
                scenario = scenario_class(self.world,
                                          self.ego_vehicles,
                                          config,
                                          self._args.randomize,
                                          self._args.debug)
        except Exception as exception:  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return False

        try:
            if self._args.record:
                recorder_name = "{}/{}/{}.log".format(
                    os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.record, config.name)
                self.client.start_recorder(recorder_name, True)

            # Load scenario and run it
            self.manager.load_scenario(scenario, self.agent_instance)
            self.manager.run_scenario()

            # Provide outputs if required
            self._analyze_scenario(config)

            # Remove all actors, stop the recorder and save all criterias (if needed)
            scenario.remove_all_actors()
            if self._args.record:
                self.client.stop_recorder()
                self._record_criteria(self.manager.scenario.get_criteria(), recorder_name)

            result = True

        except Exception as e:  # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self._cleanup()
        return result

    def store_results(self, single_result):
        """
        Store results to excel file.
        """

        # f = xlwt.Workbook()
        # sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
        #
        # sheet1.write(count_sum - 1, 4, success_num)
        # f.save('result_v_33.xlsx')
        #
        # success_num = count_sum - count_failure
        # print(count_failure)
        # print(success_num)
        #
        # sheet1.write(count_sum, 0, self.velocity)  # 记录环境车的速度
        # sheet1.write(count_sum, 1, self.distance)  # 记录环境车之间距离
        # sheet1.write(count_sum, 2, self.flag_1)  # 记录是否碰撞
        # sheet1.write(count_sum, 3, self.timestamp)  # 记录运行的时间
        #
        # sheet1.write(count_sum - 1, 4, success_num)  # 记录成功的次数
        # f.save('result1.xlsx')
        #

        print('result is stored.')

    def run_single_scenario(self, config, vel, dist, rep):
        """
        Modified based on original load_and_run_scenario.

        Run single scenario test once.

        Load and run the scenario given by config
        """
        # original flag to identify if the scenario successfully ran
        result = False

        # additional result storage
        result_flags = ['success', 'collision', 'time_exceed']
        single_result = {
            'result': None,
            'duration_time': None,  # float, duration time of single scenario test run
        }

        # original method, get prepared
        if not self._load_and_wait_for_world(config.town, config.ego_vehicles):
            self._cleanup()
            return False

        # prepare agent
        if self._args.agent:
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            try:
                # todo init agent instance with custom args
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig)
                config.agent = self.agent_instance
            except Exception as e:  # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return False

        # ============================================================
        # --------------------  modifications  --------------------
        # ============================================================

        # -----------------  retrieve scenario info  -----------------
        #
        route_flag_index = {
            'left': -1,
            'straight': 0,
            'right': 1,

            'straight_1': 0,
        }

        route_config = config.name.split('-')
        ego_route_option = route_config[1]
        traffic_route_option = route_config[3]

        # # an exception for straight scenario
        # if ego_route_option == 'straight_1':
        #     ego_route_option = 'straight'

        # get turn flag
        ego_turn_flag = route_flag_index[ego_route_option]
        traffic_turn_flag = route_flag_index[traffic_route_option]

        # -----------------  Get ego route  -----------------
        # get start location of ego vehicle from config
        route_manager = ScenarioRouteManager(world=self.world,
                                             spawn_location=config.ego_vehicles[0].transform.location,
                                             debug=True,
                                             verbose=False,
                                             )
        # get target junction
        junction = route_manager.junction
        # ego route
        # distance to drive after the junction area
        route_distance = 3.
        ego_wp_route, ego_loc_route, ego_trans_route = \
            route_manager.get_route(spawn_location=config.ego_vehicles[0].transform.location,
                                    distance=route_distance,
                                    turning_flag=ego_turn_flag,
                                    )

        tf_distance = 50.  #
        tf_wp_route, tf_loc_route, tf_trans_route = \
            route_manager.get_route(spawn_location=config.other_actors[0].transform.location,
                                    distance=tf_distance,
                                    turning_flag=traffic_turn_flag,
                                    )

        # visualize ego route
        for wp, _ in ego_wp_route:
            draw_waypoint(self.world, wp, color=(green, green))

        # visualize traffic flow route
        for wp, _ in tf_wp_route:
            draw_waypoint(self.world, wp, color=(magenta, magenta))

        # # todo fix check api to check if test agent is loaded correctly
        # if self.agent_instance.__class__.__name__ == 'TestAgent':

        # set additional API for agent instance
        try:
            # set route option for agent instance
            self.agent_instance.load_model(route_option=ego_route_option,
                                           manual=False,  # manually set model path in test_agent.py
                                           # debug=True,
                                           debug=False,
                                           )
            # set carla.Client to init state manager
            self.agent_instance.set_client(self.client, self._args.trafficManagerPort)
            # set junction for the agent state manager
            self.agent_instance.set_junction(junction)
            # set transform route for ego vehicle
            self.agent_instance.set_route(ego_trans_route)
            # reset buffer at beginning of each episode
            self.agent_instance.reset_buffer()

        except:
            print()
            raise RuntimeError('Fail to assign critical API to agent instance, please check.')

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            self._prepare_ego_vehicles(config.ego_vehicles)

            # --------------------  modifications  --------------------
            # assign ego vehicle for agent instance, init a controller
            self.agent_instance.set_ego_vehicles(self.ego_vehicles[0],
                                                 controller_timestep=1.0 / self.frame_rate,
                                                 )
            # ---------------------------------------------------------

            if self._args.openscenario:
                scenario = OpenScenario(world=self.world,
                                        ego_vehicles=self.ego_vehicles,
                                        config=config,
                                        config_file=self._args.openscenario,
                                        timeout=100000)
            elif self._args.route:
                scenario = RouteScenario(world=self.world,
                                         config=config,
                                         debug_mode=self._args.debug)
            else:
                scenario_class = self._get_scenario_class_or_fail(config.type)
                # if using developed scenario class
                if scenario_class.__name__.startswith('IntersectionContinuousTraffic'):
                    scenario = scenario_class(
                        world=self.world,
                        ego_vehicles=self.ego_vehicles,
                        config=config,
                        # randomize=self._args.randomize,
                        randomize=False,  # we use this arg to randomize distance gap
                        debug_mode=self._args.debug,
                        criteria_enable=True,
                        # timeout=80,  # todo fix time limit of a episode
                        # =========================
                        # additional args
                        ego_route=ego_loc_route,  # ego route for success check
                        wp_plan=tf_wp_route,
                        target_speed=vel,  # target speed of traffic flow
                        distance_gap=dist,  # distance between traffic flow spawning
                        verbose=True,  # visualize everything
                    )
                else:  # original lines
                    scenario = scenario_class(self.world,
                                              self.ego_vehicles,
                                              config,
                                              self._args.randomize,
                                              self._args.debug)

            # ================   original lines   ================
            # scenario_class = self._get_scenario_class_or_fail(config.type)
            # # todo fix the api to init self_defined scenarios
            # scenario = scenario_class(self.world,
            #                           self.ego_vehicles,
            #                           config,
            #                           self._args.randomize,
            #                           self._args.debug,
            #                           ego_route=wp_route,
            #                           )

        except Exception as exception:  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return False

        try:
            if self._args.record:
                recorder_name = "{}/{}/{}.log".format(
                    os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.record, config.name)
                self.client.start_recorder(recorder_name, True)

            # Load scenario and run it
            self.manager.load_scenario(scenario, self.agent_instance)

            # pre run the scenario
            min_waiting_time = 100. / vel * 3.6  # assuming that first vehicle drive 100m to cross the junction
            waiting_time = float(random.uniform(min_waiting_time, 1.25 * min_waiting_time))  # in seconds
            init_speed = 10.  # in km/h
            self.manager.pre_run_scenario(waiting_time=waiting_time,
                                          ego_init_speed=init_speed,  # in km/h
                                          )

            # data storage path
            # get route option
            route_config = config.name.split('-')
            ego_route_option = route_config[1]
            traffic_route_option = route_config[3]

            path_name = 'ego_{}_tf_{}'.format(ego_route_option, traffic_route_option)
            file_path = os.path.join(scenario_runner_path,
                                     'test_outputs',
                                     path_name,
                                     self.tag,
                                     'statistics')
            os.makedirs(file_path, exist_ok=True)

            file_name = 'Vel_{:n}_Dist_{}_Rep_{}.jsonl'.format(vel, dist, rep)

            # run_scenario will return the max_acc and min_ttc of this test run
            max_acc, min_ttc = self.manager.run_scenario(file_path=os.path.join(file_path, file_name), ego_init_speed=init_speed)

            # get running results
            if scenario.scenario.test_criteria[0].collision_time:
                single_result['result'] = 'collision'
                single_result['duration_time'] = scenario.scenario.test_criteria[0].collision_time
            elif self.manager.scenario_duration_game >= scenario_class.timeout - 1.:  # todo check how timeout passed to scenario class
                single_result['result'] = 'time_exceed'
                single_result['duration_time'] = scenario_class.timeout
            else:
                single_result['result'] = 'success'
                single_result['duration_time'] = self.manager.scenario_duration_game

            # add max_acc, min_ttc to result storage
            single_result['max_acc'] = max_acc
            single_result['min_ttc'] = min_ttc

            # write data at final line of json file
            with open(os.path.join(file_path, file_name), 'a') as f:
                f.write(json.dumps(
                    {
                        'duration': self.manager.scenario_duration_game,
                        'result': single_result['result'],
                        'max_acc': max_acc,
                        'min_ttc': min_ttc,
                    }
                ) + '\n')

            # Provide outputs if required
            self._analyze_scenario(config)

            # Remove all actors, stop the recorder and save all criterias (if needed)
            scenario.remove_all_actors()
            if self._args.record:
                self.client.stop_recorder()
                self._record_criteria(self.manager.scenario.get_criteria(), recorder_name)

            result = True

        except Exception as e:  # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self._cleanup()
        return result, single_result

    def run_group_scenarios(self, scenario_configurations, ):
        """
        Discretize params to generate scenario instances.

        Run single scenario test via load_and_run_scenario.
        """
        count_sum = 0
        count_success = 0
        count_time_exceed = 0
        count_collision = 0

        # tf vehicle speed range
        velocity_list = list(np.arange(self._args.speed_range[0],
                                       self._args.speed_range[1] + self._args.speed_range[2],
                                       self._args.speed_range[2])
                             )

        # tf vehicle gap distance range
        distance_list = list(np.arange(self._args.distance_range[0],
                                       self._args.distance_range[1] + self._args.distance_range[2],
                                       self._args.distance_range[2])
                             )

        # iter the running loop
        for dist in distance_list:
            for vel in velocity_list:

                # todo add filter for undesired scenarios
                # ie. use a dict to store filtered scenarios

                print(
                    '\n',
                    '-'*50, '\n',
                    'Running scenario:', 'Vel: ', vel, 'Dist: ', dist, '\n',
                )

                # repeat multiple times for each scenario
                for rep in range(int(self._args.repetitions)):
                    # todo remove different configurations
                    for config in scenario_configurations:
                        # get route option
                        route_config = config.name.split('-')
                        ego_route_option = route_config[1]
                        traffic_route_option = route_config[3]

                        path_name = 'ego_{}_tf_{}'.format(ego_route_option, traffic_route_option)
                        result_path = os.path.join(scenario_runner_path,
                                                   'test_outputs',
                                                   path_name,
                                                   self.tag,
                                                   )
                        os.makedirs(result_path, exist_ok=True)
                        # use pickle to store data
                        # name = 'test_result.pkl'
                        # use jsonl to store data
                        name = 'test_results.jsonl'

                        result, single_result = self.run_single_scenario(config, vel, dist, rep)
                        # store data
                        count_sum += 1
                        if single_result['result'] == 'success':
                            count_success += 1
                        elif single_result['result'] == 'time_exceed':
                            count_time_exceed += 1
                        else:  # collision
                            count_collision += 1
                        duration_time = single_result['duration_time']

                        result_dict = {
                            'velocity': float(vel),
                            'distance': float(dist),
                            'result': single_result['result'],
                            'duration': duration_time,
                            # max acc and min ttc
                            'max_acc': single_result['max_acc'],
                            'min_ttc': single_result['min_ttc'],
                        }

                        # # use pickle to store data
                        # with open(os.path.join(result_path, name), "wb") as f:
                        #     pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)

                        # use jsonl
                        with open(os.path.join(result_path, name), 'a') as f:
                            f.write(json.dumps(result_dict) + '\n')

                        # cleanup
                        tf.reset_default_graph()  # todo fix tf warning
                        self._cleanup()

        # print results
        print(
            '\n',
            '=' * 60, '\n',
            'Scenario: ', scenario_configurations[0].name, '\n',
            'Results: ', '\n',
            'Speed Range: ', self._args.speed_range, '\n',
            'Distance Range: ', self._args.distance_range, '\n',
            'Total Test Number: ', count_sum, '\n',
            'Success Number: ', count_success, '\n',
            'Collision Number: ', count_collision, '\n',
            'Time exceed Number: ', count_time_exceed, '\n',
            'Success rate: {:.2%}'.format(count_success / count_sum), '\n',
            '=' * 60,
        )

        # todo fix return results
        return True

    def _run_scenarios(self):
        """
        Run conventional scenarios (e.g. implemented using the Python API of ScenarioRunner)
        """
        result = False

        # Load the scenario configurations provided in the config file
        scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(
            self._args.scenario,
            self._args.configFile)
        if not scenario_configurations:
            print("Configuration for scenario {} cannot be found!".format(self._args.scenario))
            return result

        # # =====  original lines  =====
        # # Execute each configuration
        # for config in scenario_configurations:
        #     for _ in range(self._args.repetitions):
        #         result = self._load_and_run_scenario(config)
        #
        #     self._cleanup()
        # # ============================

        # Execute each configuration
        result = self.run_group_scenarios(scenario_configurations)

        return result

    def _run_route(self):
        """
        Run the route scenario
        """
        result = False

        if self._args.route:
            routes = self._args.route[0]
            scenario_file = self._args.route[1]
            single_route = None
            if len(self._args.route) > 2:
                single_route = self._args.route[2]

        # retrieve routes
        route_configurations = RouteParser.parse_routes_file(routes, scenario_file, single_route)

        for config in route_configurations:
            for _ in range(self._args.repetitions):
                result = self._load_and_run_scenario(config)

                self._cleanup()
        return result

    def _run_openscenario(self):
        """
        Run a scenario based on OpenSCENARIO
        """

        # Load the scenario configurations provided in the config file
        if not os.path.isfile(self._args.openscenario):
            print("File does not exist")
            self._cleanup()
            return False

        config = OpenScenarioConfiguration(self._args.openscenario, self.client)

        result = self._load_and_run_scenario(config)
        self._cleanup()
        return result

    def run(self):
        """
        Run all scenarios according to provided commandline args
        """
        result = True
        if self._args.openscenario:
            result = self._run_openscenario()
        elif self._args.route:
            result = self._run_route()
        else:
            result = self._run_scenarios()

        print("No more scenarios .... Exiting")
        return result


def main():
    """
    main function
    """
    description = ("CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n"
                   "Current version: " + VERSION)

    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--timeout', default="10.0",
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--trafficManagerPort', default='8000', type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0', type=int,
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--sync', action='store_true',
                        help='Forces the simulation to run synchronously')
    parser.add_argument('--list', action="store_true", help='List all supported scenarios and exit')

    parser.add_argument(
        '--scenario',
        help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    parser.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')
    parser.add_argument(
        '--route', help='Run a route as a scenario (input: (route_file,scenario_file,[route id]))', nargs='+', type=str)

    parser.add_argument(
        '--agent',
        default='',
        help="Agent used to execute the scenario. Currently only compatible with route-based scenarios.")
    parser.add_argument('--agentConfig', type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument('--output', action="store_true", help='Provide results on stdout')
    parser.add_argument('--file', action="store_true", help='Write results into a txt file')
    parser.add_argument('--junit', action="store_true", help='Write results into a junit file')
    parser.add_argument('--json', action="store_true", help='Write results into a JSON file')
    parser.add_argument('--outputDir', default='', help='Directory for output files (default: this directory)')

    parser.add_argument('--configFile', default='', help='Provide an additional scenario configuration file (*.xml)')
    parser.add_argument('--additionalScenario', default='', help='Provide additional scenario implementations (*.py)')

    parser.add_argument('--debug', action="store_true", help='Run with debug output')
    parser.add_argument('--reloadWorld', action="store_true",
                        help='Reload the CARLA world before starting a scenario (default=True)')
    parser.add_argument('--record', type=str, default='',
                        help='Path were the files will be saved, relative to SCENARIO_RUNNER_ROOT.\nActivates the CARLA recording feature and saves to file all the criteria information.')
    parser.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    parser.add_argument('--repetitions', default=1, type=int, help='Number of scenario executions')
    parser.add_argument('--waitForEgo', action="store_true", help='Connect the scenario to an existing ego vehicle')

    # additional args
    parser.add_argument('--speed-range', dest='speed_range', nargs='+', type=float, default=(10, 34, 2),
                        help='Target speed range of traffic flow in km/h, (lower_limit, upper_limit, step_length)')
    parser.add_argument('--distance-range', dest='distance_range', nargs='+', type=float, default=(16, 40, 2),
                        help='Distance gap range of traffic flow in meters, (lower_limit, upper_limit, step_length)')
    parser.add_argument('--tag', dest='tag', type=str, default=None,
                        help='Additional tag to identify the test experiments.')

    arguments = parser.parse_args()
    # pylint: enable=line-too-long

    if arguments.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(arguments.configFile), sep='\n')
        return 1

    if not arguments.scenario and not arguments.openscenario and not arguments.route:
        print("Please specify either a scenario or use the route mode\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.route and (arguments.openscenario or arguments.scenario):
        print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
        parser.print_help(sys.stdout)
        return 1

    # todo this line is fixed, a route will be manually added to agent
    # if arguments.agent and (arguments.openscenario or arguments.scenario):
    #     print("Agents are currently only compatible with route scenarios'\n\n")
    #     parser.print_help(sys.stdout)
    #     return 1

    if arguments.agent and (arguments.openscenario or arguments.scenario):
        print('Warning! A route is supposed to be set to the agent manually.')

    if arguments.route:
        arguments.reloadWorld = True

    if arguments.agent:
        arguments.sync = True

    scenario_runner = None
    result = True
    try:
        scenario_runner = ScenarioRunner(arguments)
        result = scenario_runner.run()

    finally:
        if scenario_runner is not None:
            scenario_runner.destroy()
            del scenario_runner
    return not result


if __name__ == "__main__":
    sys.exit(main())

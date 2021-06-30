"""
Fixed version of scenario manager.
"""

import carla

import numpy as np
import json
import sys
import time
import math

import py_trees

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.drl_code.scenario_utils.kinetics import get_speed, set_vehicle_speed


class ScenarioManagerFixed(ScenarioManager):

    def __init__(self,
                 debug_mode=False,
                 sync_mode=False,
                 timeout=2.0,
                 # ========  additional args  ========
                 sync_tick=False,
                 ):
        
        super(ScenarioManagerFixed, self).__init__(debug_mode=debug_mode, sync_mode=sync_mode, timeout=timeout)

        # if using sync tick mode
        self.sync_tick = sync_tick
        self.simulation_time = None

        # this is the original agent instance
        self.original_agent = None

    def load_scenario(self, scenario, agent=None):
        """
        Load a new scenario
        """
        self._reset()
        self._agent = AgentWrapper(agent) if agent else None
        if self._agent is not None:
            self._sync_mode = True

        # original agent
        self.original_agent = agent

        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        if self._agent is not None:
            self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def _pre_run_tick_scenario(self, timestamp, i):
        """
        Run next tick of scenario and the agent.
        If running synchornously, it also handles the ticking of the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            if i < 50:
                control = carla.VehicleControl()
                control.brake = 0.0
                control.throttle = 1.0
                control.steer = 0.0
                self.ego_vehicles[0].apply_control(control)
            else:
                control = carla.VehicleControl()
                control.brake = 1.0
                control.throttle = 0.0
                control.steer = 0.0
                self.ego_vehicles[0].apply_control(control)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

        if self._sync_mode and self._running and self._watchdog.get_status():
            if self.sync_tick:
                self.sync_tick_world()
            else:
                CarlaDataProvider.get_world().tick()

    def pre_run_scenario(self, waiting_time=30., ego_init_speed=0.):
        """
        This method is responsible to pre-run scenario before the time counting.
        Therefore to generate a well spawned traffic flow.

        This method is supposed to be called after load_scenario.
        """
        # fixme check if need to apply a brake control to ego

        self._reset()

        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        # CarlaDataProvider.get_world()
        # default timestep is 0.05
        count = waiting_time / 0.05  # todo get world timestep
        for i in range(int(count)):

            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                # self._tick_scenario(timestamp)
                self._pre_run_tick_scenario(timestamp, i)

            # self.debug_distance()

        self._watchdog.stop()

        # self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        # if self.scenario_tree.status == py_trees.common.Status.FAILURE:
        #     print("ScenarioManager: Terminated due to failure")

        self._reset()

        # set initial speed for ego vehicle
        set_vehicle_speed(self.ego_vehicles[0], ego_init_speed / 3.6)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent.
        If running synchornously, it also handles the ticking of the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            if self._agent is not None:
                ego_action = self._agent()

            if self._agent is not None:
                self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

        if self._sync_mode and self._running and self._watchdog.get_status():
            if self.sync_tick:
                self.sync_tick_world()
            else:
                CarlaDataProvider.get_world().tick()

    def run_scenario(self, file_path, ego_init_speed=0.):
        """
        Trigger the start of the scenario and wait for it to finish/fail.

        Notice! input arg of this method is fixed.
        """

        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        # init data storage
        world = CarlaDataProvider.get_world()
        snapshot = world.get_snapshot()
        timestamp = snapshot.timestamp

        start_frame = timestamp.frame
        last_time = timestamp.elapsed_seconds
        last_speed = ego_init_speed / 3.6
        max_acc = 0.
        min_ttc = 1e9  # minimum ttc

        # this is the main loop of scenario running
        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

            # debug method
            # self.debug_distance()

            # get ttc from state_manager5
            ttc = None
            try:
                ttc = self.original_agent.state_manager.get_ttc()
                print('TTC: ', ttc)
            except:
                print('Fail to get ttc value, please check agent class method.')

            # if tick the world successfully, record data
            # timestamp is updated!
            world = CarlaDataProvider.get_world()
            snapshot = world.get_snapshot()
            timestamp = snapshot.timestamp
            frame, elapsed_seconds = timestamp.frame, timestamp.elapsed_seconds

            if (frame - start_frame) % 10 == 0:
                current_time = GameTime.get_carla_time()
                current_speed = get_speed(self.ego_vehicles[0])

                # delta_time = (current_time - last_time)

                acc = (current_speed - last_speed) / (current_time - last_time)
                abs_acc = math.fabs(acc)
                max_acc = max(abs_acc, max_acc)

                with open(file_path, 'a') as f:
                    f.write(json.dumps(
                        {
                            'timestamp': current_time,
                            'current_speed': current_speed,
                            'last_speed': last_speed,
                            'delta_time': (current_time - last_time),
                            'abs_acc': abs_acc,
                            'max_acc': max_acc,
                            'min_ttc': min_ttc,
                        }
                    ) + '\n')

                print("current_speed:{:.3}, last_speed:{:.3}, acc:{:.3}".format(current_speed, last_speed, abs_acc))

                # update data when acc is updated
                last_time = current_time
                last_speed = current_speed

            else:
                # info need to be stored every timestep
                current_time = GameTime.get_carla_time()
                # scalar speed in m/s
                current_speed = get_speed(self.ego_vehicles[0])
                # minimum ttc of scenario
                min_ttc = min(min_ttc, ttc)

                # retrieve additional info from test agent
                try:
                    state, action = self.original_agent.get_step_info()
                    state = state.tolist()
                    action = action.tolist()
                    data_dict = {
                        'timestamp': current_time,
                        'current_speed': current_speed,
                        'ttc': ttc,
                        'state': state,
                        'action': action,
                    }
                except:
                    data_dict = {
                        'timestamp': current_time,
                        'current_speed': current_speed,
                        'ttc': ttc,
                    }
                    print('Current agent doesn\'t have such API.')

                # record data
                with open(file_path, 'a') as f:
                    f.write(json.dumps(data_dict) + '\n')

        self._watchdog.stop()

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

        print(
            '-' * 50, '\n',
            'Max Acceleration: ', max_acc, 'm^2/s', '\n',
            'Min TTC: ', min_ttc, 's', '\n',
            '-' * 50,
        )

        return max_acc, min_ttc

    def sync_tick_world(self):
        """
        Tick carla world in sync mode.

        To ensure that wall clock time step is fixed.
        """
        CarlaDataProvider.get_world().tick()

        if not self.simulation_time:
            self.simulation_time = time.time()

        current_time = time.time()
        elapsed = current_time - self.simulation_time

        # todo retrieve timestep of the world
        # if elapsed < self.server_dt:
        #     time.sleep(self.server_dt - elapsed)
        server_dt = 0.05
        if elapsed < server_dt:
            time.sleep(server_dt - elapsed)

        self.simulation_time = time.time()

    def debug_distance(self):
        """
        Fix the error of distance between npc vehicles
        Returns:
        """
        # check speed of traffic flow vehicles
        # world = self._world
        world = CarlaDataProvider.get_world()

        npc_vehicles, ego_vehicle = self.update_vehicles(world)
        # check speed
        for veh in npc_vehicles:
            speed = 3.6 * get_speed(veh)  # in km/h
            _location = veh.get_location()
            location = [_location.x, _location.y, _location.z]
            print('Vehicle {}, speed: {}, location: {}'.format(veh.id, speed, location))

        print('')

    @staticmethod
    def update_vehicles(world):
        """
        static method of update vehicles.
        """
        npc_vehicles = []
        ego_vehicle = None

        actor_list = world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')  # carla.Actorlist, iterable

        if vehicle_list:
            for veh in vehicle_list:  # veh is carla.Vehicle
                attr = veh.attributes  # dict
                role_name = attr['role_name']
                # filter ego vehicle
                if role_name in ['ego', 'hero']:
                    ego_vehicle = veh
                else:
                    npc_vehicles.append(veh)

        return npc_vehicles, ego_vehicle

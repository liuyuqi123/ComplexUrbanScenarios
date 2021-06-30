"""
A developing version of traffic lights manager.

The desired new functions are:

 - a arg to switch on/off traffic lights, which means
    True refers to enable traffic lights in the specified junction,
    False refers to shut all traffic lights by freeze its state to Green

"""

import carla

import numpy as np

from train.gym_carla.modules.traffic_lights.traffic_lights_manager import TrafficLightsController


class TrafficLightsManager(TrafficLightsController):

    # phase_time refers to the duration time of each state of traffic lights on x direction
    # notice the format
    phase_time = {
        'x_green_phase': 12.,  # 19.
        'x_yellow_phase': 3.,
        'y_green_phase': 27.,
        'y_yellow_phase': 3.,
    }

    def __init__(self,
                 carla_api,
                 junction: carla.Junction = None,
                 use_tls_control=True,
                 ):

        super(TrafficLightsManager, self).__init__(carla_api=carla_api,
                                                   junction=junction,
                                                   )

        # get traffic lights in current junction
        self.get_traffic_lights()
        # whether use tls logic
        self.use_tls_control = use_tls_control

        # turnoff tls control
        if not self.use_tls_control:
            self.turnoff_tls_control()
            print('Traffic lights control is disabled.')
        else:
            print('Traffic lights control is enabled.')

    def switch_tls_control(self, use_tls_control: bool):
        """
        Switch on or off traffic lights.
        """
        self.use_tls_control = use_tls_control

        if self.use_tls_control:  # switch on: reset
            self.set_tl_state()
        else:  # switch off: set to green
            self.turnoff_tls_control()

    def turnoff_tls_control(self):
        """
        Set all traffic lights in specified junction to Green permanently.
        :return:
        """
        # check tl exist
        if not self.traffic_lights:
            raise RuntimeError('Traffic lights are not found!')

        # clear the elapsed time
        self.traffic_lights[0].reset_group()

        for tl in self.traffic_lights:
            tl.set_green_time(999999.)
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        # requires manual tick after setting state(in sync mode)
        self.world.tick()

    def run_step(self):
        """
        This method will be ticked each timestep with the RL env.

        This method is supposed to be called after world tick.
        """
        # if tls control is disabled, do nothing
        if not self.use_tls_control:
            return

        # get current timestamp
        timestamp = self.world.get_snapshot().timestamp
        t_1 = timestamp.elapsed_seconds  # current time

        # duration time of each state
        t_0 = self.last_change_timestamp.elapsed_seconds  # latest time

        # elapsed time since last change
        elapsed_time = t_1 - t_0

        # current information of the
        # tl_info = self.get_tl_info()

        # duration time of current phase
        phase_name = 'phase_' + str(self.current_phase_index)
        phase_duration = self.tl_logic[phase_name]['duration']

        # if time is reached
        if elapsed_time >= phase_duration:
            # shift to next state
            if self.current_phase_index == int(3):
                self.current_phase_index = 0
            else:
                self.current_phase_index = int(self.current_phase_index + 1)

            # set the traffic light to next phase
            self.set_tl_state(self.current_phase_index)

            # reset timer
            self.last_change_timestamp = timestamp


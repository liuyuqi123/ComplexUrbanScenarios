"""
Sensors for vehicles and other actors.
"""

import carla


class Sensors(object):
    """
    todo fix api for optimization

    Collision sensor for vehicles.
    """
    def __init__(self, world, ego_car):
        super(Sensors, self).__init__()
        self.world = world
        self.ego_car = ego_car
        # self.camera_queue = queue.Queue() # queue to store images from buffer
        self.collision_flag = False  # Flag for colision detection
        self.collision_type = ''  # Which type of actor was crashed
        # self.lane_crossed = False # Flag for lane crossing detection
        # self.lane_crossed_type = '' # Which type of lane was crossed

        # self.camera_rgb = self.add_sensors(world, ego_car, 'sensor.camera.rgb')
        self.collision = self.add_sensors(world, ego_car, 'sensor.other.collision')
        # self.lane_invasion = self.add_sensors(world, ego_car, 'sensor.other.lane_invasion') # sensor_tick = '0.5'
        # todo sensor list contains other kinds of sensors
        self.sensor_list = [self.collision]  # self.lane_invasion, self.camera_rgb

        self.collision.listen(lambda collisionEvent: self.track_collision(collisionEvent))
        # self.camera_rgb.listen(lambda image: self.camera_queue.put(image))
        # self.lane_invasion.listen(lambda event: self.on_invasion(event))

    def add_sensors(self, world, ego_car, type, sensor_tick='0.0'):
        sensor_bp = self.world.get_blueprint_library().find(type)
        try:
            sensor_bp.set_attribute('sensor_tick', sensor_tick)
        except:
            # print('Fail to add sensor attribute.')
            pass
        # if type == 'sensor.camera.rgb':
        #     sensor_bp.set_attribute('image_size_x', '100')
        #     sensor_bp.set_attribute('image_size_y', '100')

        sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=ego_car)
        return sensor

    def track_collision(self, collisionEvent):
        '''Whenever a collision occurs, the flag is set to True'''
        actor_types = collisionEvent.other_actor.type_id
        self.collision_type = actor_types.split('.')[0]
        self.collision_flag = True

    def reset_sensors(self):
        '''Sets all sensor flags to False'''
        self.collision_flag = False
        # self.lane_crossed = False
        # self.lane_crossed_type = ''

    def on_invasion(self, event):
        '''Whenever the car crosses the lane, the flag is set to True'''
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.lane_crossed_type = text[0]
        self.lane_crossed = True

    def destroy_sensors(self):
        '''Destroy all sensors (Carla actors)'''
        # client.apply_batch([carla.command.DestroyActor(sensor) for sensor in self.sensor_list])
        for sensor in self.sensor_list:
            sensor.destroy()

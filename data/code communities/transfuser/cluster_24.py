# Cluster 24

class ActorTransformSetterToOSCPosition(AtomicBehavior):
    """
    OpenSCENARIO atomic
    This class contains an atomic behavior to set the transform of an OpenSCENARIO actor.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - osc_position: OpenSCENARIO position
    - physics [optional]: If physics is true, the actor physics will be reactivated upon success

    The behavior terminates when actor is set to the new actor transform (closer than 1 meter)

    NOTE:
    It is very important to ensure that the actor location is spawned to the new transform because of the
    appearence of a rare runtime processing error. WaypointFollower with LocalPlanner,
    might fail if new_status is set to success before the actor is really positioned at the new transform.
    Therefore: calculate_distance(actor, transform) < 1 meter
    """

    def __init__(self, actor, osc_position, physics=True, name='ActorTransformSetterToOSCPosition'):
        """
        Setup parameters
        """
        super(ActorTransformSetterToOSCPosition, self).__init__(name, actor)
        self._osc_position = osc_position
        self._physics = physics
        self._osc_transform = None

    def initialise(self):
        super(ActorTransformSetterToOSCPosition, self).initialise()
        if self._actor.is_alive:
            self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

    def update(self):
        """
        Transform actor
        """
        new_status = py_trees.common.Status.RUNNING
        self._osc_transform = srunner.tools.openscenario_parser.OpenScenarioParser.convert_position_to_transform(self._osc_position)
        self._actor.set_transform(self._osc_transform)
        if not self._actor.is_alive:
            new_status = py_trees.common.Status.FAILURE
        if calculate_distance(self._actor.get_location(), self._osc_transform.location) < 1.0:
            if self._physics:
                self._actor.set_simulate_physics(enabled=True)
            new_status = py_trees.common.Status.SUCCESS
        return new_status

def calculate_distance(location, other_location, global_planner=None):
    """
    Method to calculate the distance between to locations

    Note: It uses the direct distance between the current location and the
          target location to estimate the time to arrival.
          To be accurate, it would have to use the distance along the
          (shortest) route between the two locations.
    """
    if global_planner:
        distance = 0
        route = global_planner.trace_route(location, other_location)
        for i in range(1, len(route)):
            curr_loc = route[i][0].transform.location
            prev_loc = route[i - 1][0].transform.location
            distance += curr_loc.distance(prev_loc)
        return distance
    return location.distance(other_location)

class KeepVelocity(AtomicBehavior):
    """
    This class contains an atomic behavior to keep the provided velocity.
    The controlled traffic participant will accelerate as fast as possible
    until reaching a given _target_velocity_, which is then maintained for
    as long as this behavior is active.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_velocity: The target velocity the actor should reach
    - duration[optional]: Duration in seconds of this behavior
    - distance[optional]: Maximum distance in meters covered by the actor during this behavior

    A termination can be enforced by providing distance or duration values.
    Alternatively, a parallel termination behavior has to be used.
    """

    def __init__(self, actor, target_velocity, duration=float('inf'), distance=float('inf'), name='KeepVelocity'):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(KeepVelocity, self).__init__(name, actor)
        self.logger.debug('%s.__init__()' % self.__class__.__name__)
        self._target_velocity = target_velocity
        self._control, self._type = get_actor_control(actor)
        self._map = self._actor.get_world().get_map()
        self._waypoint = self._map.get_waypoint(self._actor.get_location())
        self._duration = duration
        self._target_distance = distance
        self._distance = 0
        self._start_time = 0
        self._location = None

    def initialise(self):
        self._location = CarlaDataProvider.get_location(self._actor)
        self._start_time = GameTime.get_time()
        if self._type == 'walker':
            self._control.speed = self._target_velocity
            self._control.direction = CarlaDataProvider.get_transform(self._actor).get_forward_vector()
        super(KeepVelocity, self).initialise()

    def update(self):
        """
        As long as the stop condition (duration or distance) is not violated, set a new vehicle control

        For vehicles: set throttle to throttle_value, as long as velocity is < target_velocity
        For walkers: simply apply the given self._control
        """
        new_status = py_trees.common.Status.RUNNING
        if self._type == 'vehicle':
            if CarlaDataProvider.get_velocity(self._actor) < self._target_velocity:
                self._control.throttle = 1.0
            else:
                self._control.throttle = 0.0
        self._actor.apply_control(self._control)
        new_location = CarlaDataProvider.get_location(self._actor)
        self._distance += calculate_distance(self._location, new_location)
        self._location = new_location
        if self._distance > self._target_distance:
            new_status = py_trees.common.Status.SUCCESS
        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS
        self.logger.debug('%s.update()[%s->%s]' % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        if self._type == 'vehicle':
            self._control.throttle = 0.0
        elif self._type == 'walker':
            self._control.speed = 0.0
        if self._actor is not None and self._actor.is_alive:
            self._actor.apply_control(self._control)
        super(KeepVelocity, self).terminate(new_status)

class SyncArrival(AtomicBehavior):
    """
    This class contains an atomic behavior to
    set velocity of actor so that it reaches location at the same time as
    actor_reference. The behavior assumes that the two actors are moving
    towards location in a straight line.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - actor_reference: Reference actor with which arrival is synchronized
    - target_location: CARLA location where the actors should "meet"
    - gain[optional]: Coefficient for actor's throttle and break controls

    Note: In parallel to this behavior a termination behavior has to be used
          to keep continue synchronization for a certain duration, or for a
          certain distance, etc.
    """

    def __init__(self, actor, actor_reference, target_location, gain=1, name='SyncArrival'):
        """
        Setup required parameters
        """
        super(SyncArrival, self).__init__(name, actor)
        self.logger.debug('%s.__init__()' % self.__class__.__name__)
        self._control = carla.VehicleControl()
        self._actor_reference = actor_reference
        self._target_location = target_location
        self._gain = gain
        self._control.steering = 0

    def update(self):
        """
        Dynamic control update for actor velocity
        """
        new_status = py_trees.common.Status.RUNNING
        distance_reference = calculate_distance(CarlaDataProvider.get_location(self._actor_reference), self._target_location)
        distance = calculate_distance(CarlaDataProvider.get_location(self._actor), self._target_location)
        velocity_reference = CarlaDataProvider.get_velocity(self._actor_reference)
        time_reference = float('inf')
        if velocity_reference > 0:
            time_reference = distance_reference / velocity_reference
        velocity_current = CarlaDataProvider.get_velocity(self._actor)
        time_current = float('inf')
        if velocity_current > 0:
            time_current = distance / velocity_current
        control_value = self._gain * (time_current - time_reference)
        if control_value > 0:
            self._control.throttle = min([control_value, 1])
            self._control.brake = 0
        else:
            self._control.throttle = 0
            self._control.brake = min([abs(control_value), 1])
        self._actor.apply_control(self._control)
        self.logger.debug('%s.update()[%s->%s]' % (self.__class__.__name__, self.status, new_status))
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior, the throttle should be set back to 0.,
        to avoid further acceleration.
        """
        if self._actor is not None and self._actor.is_alive:
            self._control.throttle = 0.0
            self._control.brake = 0.0
            self._actor.apply_control(self._control)
        super(SyncArrival, self).terminate(new_status)

class BasicAgentBehavior(AtomicBehavior):
    """
    This class contains an atomic behavior, which uses the
    basic_agent from CARLA to control the actor until
    reaching a target location.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - target_location: Is the desired target location (carla.location),
                       the actor should move to

    The behavior terminates after reaching the target_location (within 2 meters)
    """
    _acceptable_target_distance = 2

    def __init__(self, actor, target_location, name='BasicAgentBehavior'):
        """
        Setup actor and maximum steer value
        """
        super(BasicAgentBehavior, self).__init__(name, actor)
        self.logger.debug('%s.__init__()' % self.__class__.__name__)
        self._agent = BasicAgent(actor)
        self._agent.set_destination((target_location.x, target_location.y, target_location.z))
        self._control = carla.VehicleControl()
        self._target_location = target_location

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        self._control = self._agent.run_step()
        location = CarlaDataProvider.get_location(self._actor)
        if calculate_distance(location, self._target_location) < self._acceptable_target_distance:
            new_status = py_trees.common.Status.SUCCESS
        self.logger.debug('%s.update()[%s->%s]' % (self.__class__.__name__, self.status, new_status))
        self._actor.apply_control(self._control)
        return new_status

    def terminate(self, new_status):
        self._control.throttle = 0.0
        self._control.brake = 0.0
        self._actor.apply_control(self._control)
        super(BasicAgentBehavior, self).terminate(new_status)

class ActorTransformSetter(AtomicBehavior):
    """
    This class contains an atomic behavior to set the transform
    of an actor.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - transform: New target transform (position + orientation) of the actor
    - physics [optional]: If physics is true, the actor physics will be reactivated upon success

    The behavior terminates when actor is set to the new actor transform (closer than 1 meter)

    NOTE:
    It is very important to ensure that the actor location is spawned to the new transform because of the
    appearence of a rare runtime processing error. WaypointFollower with LocalPlanner,
    might fail if new_status is set to success before the actor is really positioned at the new transform.
    Therefore: calculate_distance(actor, transform) < 1 meter
    """

    def __init__(self, actor, transform, physics=True, name='ActorTransformSetter'):
        """
        Init
        """
        super(ActorTransformSetter, self).__init__(name, actor)
        self._transform = transform
        self._physics = physics
        self.logger.debug('%s.__init__()' % self.__class__.__name__)

    def initialise(self):
        if self._actor.is_alive:
            self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_transform(self._transform)
        super(ActorTransformSetter, self).initialise()

    def update(self):
        """
        Transform actor
        """
        new_status = py_trees.common.Status.RUNNING
        if not self._actor.is_alive:
            new_status = py_trees.common.Status.FAILURE
        if calculate_distance(self._actor.get_location(), self._transform.location) < 1.0:
            if self._physics:
                self._actor.set_simulate_physics(enabled=True)
            new_status = py_trees.common.Status.SUCCESS
        return new_status


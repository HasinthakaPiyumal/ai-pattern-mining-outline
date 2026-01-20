# Cluster 25

class AccelerateToVelocity(AtomicBehavior):
    """
    This class contains an atomic acceleration behavior. The controlled
    traffic participant will accelerate with _throttle_value_ until reaching
    a given _target_velocity_

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - throttle_value: The amount of throttle used to accelerate in [0,1]
    - target_velocity: The target velocity the actor should reach in m/s

    The behavior will terminate, if the actor's velocity is at least target_velocity
    """

    def __init__(self, actor, throttle_value, target_velocity, name='Acceleration'):
        """
        Setup parameters including acceleration value (via throttle_value)
        and target velocity
        """
        super(AccelerateToVelocity, self).__init__(name, actor)
        self.logger.debug('%s.__init__()' % self.__class__.__name__)
        self._control, self._type = get_actor_control(actor)
        self._throttle_value = throttle_value
        self._target_velocity = target_velocity

    def initialise(self):
        if self._type == 'walker':
            self._control.speed = self._target_velocity
            self._control.direction = CarlaDataProvider.get_transform(self._actor).get_forward_vector()
        super(AccelerateToVelocity, self).initialise()

    def update(self):
        """
        Set throttle to throttle_value, as long as velocity is < target_velocity
        """
        new_status = py_trees.common.Status.RUNNING
        if self._type == 'vehicle':
            if CarlaDataProvider.get_velocity(self._actor) < self._target_velocity:
                self._control.throttle = self._throttle_value
            else:
                new_status = py_trees.common.Status.SUCCESS
                self._control.throttle = 0
        self._actor.apply_control(self._control)
        self.logger.debug('%s.update()[%s->%s]' % (self.__class__.__name__, self.status, new_status))
        return new_status

def get_actor_control(actor):
    """
    Method to return the type of control to the actor.
    """
    control = actor.get_control()
    actor_type = actor.type_id.split('.')[0]
    if not isinstance(actor, carla.Walker):
        control.steering = 0
    else:
        control.speed = 0
    return (control, actor_type)

class AccelerateToCatchUp(AtomicBehavior):
    """
    This class contains an atomic acceleration behavior.
    The car will accelerate until it is faster than another car, in order to catch up distance.
    This behaviour is especially useful before a lane change (e.g. LaneChange atom).

    Important parameters:
    - actor: CARLA actor to execute the behaviour
    - other_actor: Reference CARLA actor, actor you want to catch up to
    - throttle_value: acceleration value between 0.0 and 1.0
    - delta_velocity: speed up to the velocity of other actor plus delta_velocity
    - trigger_distance: distance between the actors
    - max_distance: driven distance to catch up has to be smaller than max_distance

    The behaviour will terminate succesful, when the two actors are in trigger_distance.
    If max_distance is driven by the actor before actors are in trigger_distance,
    then the behaviour ends with a failure.
    """

    def __init__(self, actor, other_actor, throttle_value=1, delta_velocity=10, trigger_distance=5, max_distance=500, name='AccelerateToCatchUp'):
        """
        Setup parameters
        The target_speet is calculated on the fly.
        """
        super(AccelerateToCatchUp, self).__init__(name, actor)
        self._other_actor = other_actor
        self._throttle_value = throttle_value
        self._delta_velocity = delta_velocity
        self._trigger_distance = trigger_distance
        self._max_distance = max_distance
        self._control, self._type = get_actor_control(actor)
        self._initial_actor_pos = None

    def initialise(self):
        self._initial_actor_pos = CarlaDataProvider.get_location(self._actor)
        super(AccelerateToCatchUp, self).initialise()

    def update(self):
        actor_speed = CarlaDataProvider.get_velocity(self._actor)
        target_speed = CarlaDataProvider.get_velocity(self._other_actor) + self._delta_velocity
        distance = CarlaDataProvider.get_location(self._actor).distance(CarlaDataProvider.get_location(self._other_actor))
        driven_distance = CarlaDataProvider.get_location(self._actor).distance(self._initial_actor_pos)
        if actor_speed < target_speed:
            self._control.throttle = self._throttle_value
        if actor_speed >= target_speed:
            self._control.throttle = 0
        self._actor.apply_control(self._control)
        if distance <= self._trigger_distance:
            new_status = py_trees.common.Status.SUCCESS
        elif driven_distance > self._max_distance:
            new_status = py_trees.common.Status.FAILURE
        else:
            new_status = py_trees.common.Status.RUNNING
        return new_status

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

class StopVehicle(AtomicBehavior):
    """
    This class contains an atomic stopping behavior. The controlled traffic
    participant will decelerate with _bake_value_ until reaching a full stop.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - brake_value: Brake value in [0,1] applied

    The behavior terminates when the actor stopped moving
    """

    def __init__(self, actor, brake_value, name='Stopping'):
        """
        Setup _actor and maximum braking value
        """
        super(StopVehicle, self).__init__(name, actor)
        self.logger.debug('%s.__init__()' % self.__class__.__name__)
        self._control, self._type = get_actor_control(actor)
        if self._type == 'walker':
            self._control.speed = 0
        self._brake_value = brake_value

    def update(self):
        """
        Set brake to brake_value until reaching full stop
        """
        new_status = py_trees.common.Status.RUNNING
        if self._type == 'vehicle':
            if CarlaDataProvider.get_velocity(self._actor) > EPSILON:
                self._control.brake = self._brake_value
            else:
                new_status = py_trees.common.Status.SUCCESS
                self._control.brake = 0
        else:
            new_status = py_trees.common.Status.SUCCESS
        self._actor.apply_control(self._control)
        self.logger.debug('%s.update()[%s->%s]' % (self.__class__.__name__, self.status, new_status))
        return new_status

class WaypointFollower(AtomicBehavior):
    """
    This is an atomic behavior to follow waypoints while maintaining a given speed.
    If no plan is provided, the actor will follow its foward waypoints indefinetely.
    Otherwise, the behavior will end with SUCCESS upon reaching the end of the plan.
    If no target velocity is provided, the actor continues with its current velocity.

    Args:
        actor (carla.Actor):  CARLA actor to execute the behavior.
        target_speed (float, optional): Desired speed of the actor in m/s. Defaults to None.
        plan ([carla.Location] or [(carla.Waypoint, carla.agent.navigation.local_planner)], optional):
            Waypoint plan the actor should follow. Defaults to None.
        blackboard_queue_name (str, optional):
            Blackboard variable name, if additional actors should be created on-the-fly. Defaults to None.
        avoid_collision (bool, optional):
            Enable/Disable(=default) collision avoidance for vehicles/bikes. Defaults to False.
        name (str, optional): Name of the behavior. Defaults to "FollowWaypoints".

    Attributes:
        actor (carla.Actor):  CARLA actor to execute the behavior.
        name (str, optional): Name of the behavior.
        _target_speed (float, optional): Desired speed of the actor in m/s. Defaults to None.
        _plan ([carla.Location] or [(carla.Waypoint, carla.agent.navigation.local_planner)]):
            Waypoint plan the actor should follow. Defaults to None.
        _blackboard_queue_name (str):
            Blackboard variable name, if additional actors should be created on-the-fly. Defaults to None.
        _avoid_collision (bool): Enable/Disable(=default) collision avoidance for vehicles/bikes. Defaults to False.
        _actor_dict: Dictonary of all actors, and their corresponding plans (e.g. {actor: plan}).
        _local_planner_dict: Dictonary of all actors, and their corresponding local planners.
            Either "Walker" for pedestrians, or a carla.agent.navigation.LocalPlanner for other actors.
        _args_lateral_dict: Parameters for the PID of the used carla.agent.navigation.LocalPlanner.
        _unique_id: Unique ID of the behavior based on timestamp in nanoseconds.

    Note:
        OpenScenario:
        The WaypointFollower atomic must be called with an individual name if multiple consecutive WFs.
        Blackboard variables with lists are used for consecutive WaypointFollower behaviors.
        Termination of active WaypointFollowers in initialise of AtomicBehavior because any
        following behavior must terminate the WaypointFollower.
    """

    def __init__(self, actor, target_speed=None, plan=None, blackboard_queue_name=None, avoid_collision=False, name='FollowWaypoints'):
        """
        Set up actor and local planner
        """
        super(WaypointFollower, self).__init__(name, actor)
        self._actor_dict = {}
        self._actor_dict[actor] = None
        self._target_speed = target_speed
        self._local_planner_dict = {}
        self._local_planner_dict[actor] = None
        self._plan = plan
        self._blackboard_queue_name = blackboard_queue_name
        if blackboard_queue_name is not None:
            self._queue = Blackboard().get(blackboard_queue_name)
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}
        self._avoid_collision = avoid_collision
        self._unique_id = 0

    def initialise(self):
        """
        Delayed one-time initialization

        Checks if another WaypointFollower behavior is already running for this actor.
        If this is the case, a termination signal is sent to the running behavior.
        """
        super(WaypointFollower, self).initialise()
        self._unique_id = int(round(time.time() * 1000000000.0))
        try:
            check_attr = operator.attrgetter('running_WF_actor_{}'.format(self._actor.id))
            running = check_attr(py_trees.blackboard.Blackboard())
            active_wf = copy.copy(running)
            active_wf.append(self._unique_id)
            py_trees.blackboard.Blackboard().set('running_WF_actor_{}'.format(self._actor.id), active_wf, overwrite=True)
        except AttributeError:
            py_trees.blackboard.Blackboard().set('terminate_WF_actor_{}'.format(self._actor.id), [], overwrite=True)
            py_trees.blackboard.Blackboard().set('running_WF_actor_{}'.format(self._actor.id), [self._unique_id], overwrite=True)
        for actor in self._actor_dict:
            self._apply_local_planner(actor)
        return True

    def _apply_local_planner(self, actor):
        """
        Convert the plan into locations for walkers (pedestrians), or to a waypoint list for other actors.
        For non-walkers, activate the carla.agent.navigation.LocalPlanner module.
        """
        if self._target_speed is None:
            self._target_speed = CarlaDataProvider.get_velocity(actor)
        else:
            self._target_speed = self._target_speed
        if isinstance(actor, carla.Walker):
            self._local_planner_dict[actor] = 'Walker'
            if self._plan is not None:
                if isinstance(self._plan[0], carla.Location):
                    self._actor_dict[actor] = self._plan
                else:
                    self._actor_dict[actor] = [element[0].transform.location for element in self._plan]
        else:
            local_planner = LocalPlanner(actor, opt_dict={'target_speed': self._target_speed * 3.6, 'lateral_control_dict': self._args_lateral_dict})
            if self._plan is not None:
                if isinstance(self._plan[0], carla.Location):
                    plan = []
                    for location in self._plan:
                        waypoint = CarlaDataProvider.get_map().get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Any)
                        plan.append((waypoint, RoadOption.LANEFOLLOW))
                    local_planner.set_global_plan(plan)
                else:
                    local_planner.set_global_plan(self._plan)
            self._local_planner_dict[actor] = local_planner
            self._actor_dict[actor] = self._plan

    def update(self):
        """
        Compute next control step for the given waypoint plan, obtain and apply control to actor
        """
        new_status = py_trees.common.Status.RUNNING
        check_term = operator.attrgetter('terminate_WF_actor_{}'.format(self._actor.id))
        terminate_wf = check_term(py_trees.blackboard.Blackboard())
        check_run = operator.attrgetter('running_WF_actor_{}'.format(self._actor.id))
        active_wf = check_run(py_trees.blackboard.Blackboard())
        if self._unique_id in terminate_wf:
            terminate_wf.remove(self._unique_id)
            if self._unique_id in active_wf:
                active_wf.remove(self._unique_id)
            py_trees.blackboard.Blackboard().set('terminate_WF_actor_{}'.format(self._actor.id), terminate_wf, overwrite=True)
            py_trees.blackboard.Blackboard().set('running_WF_actor_{}'.format(self._actor.id), active_wf, overwrite=True)
            new_status = py_trees.common.Status.SUCCESS
            return new_status
        if self._blackboard_queue_name is not None:
            while not self._queue.empty():
                actor = self._queue.get()
                if actor is not None and actor not in self._actor_dict:
                    self._apply_local_planner(actor)
        success = True
        for actor in self._local_planner_dict:
            local_planner = self._local_planner_dict[actor] if actor else None
            if actor is not None and actor.is_alive and (local_planner is not None):
                if not isinstance(actor, carla.Walker):
                    control = local_planner.run_step(debug=False)
                    if self._avoid_collision and detect_lane_obstacle(actor):
                        control.throttle = 0.0
                        control.brake = 1.0
                    actor.apply_control(control)
                    if local_planner._waypoints_queue:
                        success = False
                else:
                    actor_location = CarlaDataProvider.get_location(actor)
                    success = False
                    if self._actor_dict[actor]:
                        location = self._actor_dict[actor][0]
                        direction = location - actor_location
                        direction_norm = math.sqrt(direction.x ** 2 + direction.y ** 2)
                        control = actor.get_control()
                        control.speed = self._target_speed
                        control.direction = direction / direction_norm
                        actor.apply_control(control)
                        if direction_norm < 1.0:
                            self._actor_dict[actor] = self._actor_dict[actor][1:]
                            if self._actor_dict[actor] is None:
                                success = True
                    else:
                        control = actor.get_control()
                        control.speed = self._target_speed
                        control.direction = CarlaDataProvider.get_transform(actor).rotation.get_forward_vector()
                        actor.apply_control(control)
        if success:
            new_status = py_trees.common.Status.SUCCESS
        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior,
        the controls should be set back to 0.
        """
        for actor in self._local_planner_dict:
            if actor is not None and actor.is_alive:
                control, _ = get_actor_control(actor)
                actor.apply_control(control)
                local_planner = self._local_planner_dict[actor]
                if local_planner is not None and local_planner != 'Walker':
                    local_planner.reset_vehicle()
                    local_planner = None
        self._local_planner_dict = {}
        self._actor_dict = {}
        super(WaypointFollower, self).terminate(new_status)

class HandBrakeVehicle(AtomicBehavior):
    """
    This class contains an atomic hand brake behavior.
    To set the hand brake value of the vehicle.

    Important parameters:
    - vehicle: CARLA actor to execute the behavior
    - hand_brake_value to be applied in [0,1]

    The behavior terminates after setting the hand brake value
    """

    def __init__(self, vehicle, hand_brake_value, name='Braking'):
        """
        Setup vehicle control and brake value
        """
        super(HandBrakeVehicle, self).__init__(name)
        self.logger.debug('%s.__init__()' % self.__class__.__name__)
        self._vehicle = vehicle
        self._control, self._type = get_actor_control(vehicle)
        self._hand_brake_value = hand_brake_value

    def update(self):
        """
        Set handbrake
        """
        new_status = py_trees.common.Status.SUCCESS
        if self._type == 'vehicle':
            self._control.hand_brake = self._hand_brake_value
            self._vehicle.apply_control(self._control)
        else:
            self._hand_brake_value = None
            self.logger.debug('%s.update()[%s->%s]' % (self.__class__.__name__, self.status, new_status))
            self._vehicle.apply_control(self._control)
        return new_status


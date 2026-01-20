# Cluster 29

class Entity(TorchVectorizedObject, Observable, ABC):

    def __init__(self, name: str, movable: bool=False, rotatable: bool=False, collide: bool=True, density: float=25.0, mass: float=1.0, shape: Shape=None, v_range: float=None, max_speed: float=None, color=Color.GRAY, is_joint: bool=False, drag: float=None, linear_friction: float=None, angular_friction: float=None, gravity: typing.Union[float, Tensor]=None, collision_filter: Callable[[Entity], bool]=lambda _: True):
        if shape is None:
            shape = Sphere()
        TorchVectorizedObject.__init__(self)
        Observable.__init__(self)
        self._name = name
        self._movable = movable
        self._rotatable = rotatable
        self._collide = collide
        self._density = density
        self._mass = mass
        self._max_speed = max_speed
        self._v_range = v_range
        self._color = color
        self._shape = shape
        self._is_joint = is_joint
        self._collision_filter = collision_filter
        self._state = EntityState()
        self._drag = drag
        self._linear_friction = linear_friction
        self._angular_friction = angular_friction
        if isinstance(gravity, Tensor):
            self._gravity = gravity
        else:
            self._gravity = torch.tensor(gravity, device=self.device, dtype=torch.float32) if gravity is not None else gravity
        self._goal = None
        self._render = None

    @TorchVectorizedObject.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        TorchVectorizedObject.batch_dim.fset(self, batch_dim)
        self._state.batch_dim = batch_dim

    @property
    def is_rendering(self):
        if self._render is None:
            self.reset_render()
        return self._render

    def reset_render(self):
        self._render = torch.full((self.batch_dim,), True, device=self.device)

    def collides(self, entity: Entity):
        if not self.collide:
            return False
        return self._collision_filter(entity)

    @property
    def is_joint(self):
        return self._is_joint

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, mass: float):
        self._mass = mass

    @property
    def moment_of_inertia(self):
        return self.shape.moment_of_inertia(self.mass)

    @property
    def state(self):
        return self._state

    @property
    def movable(self):
        return self._movable

    @property
    def collide(self):
        return self._collide

    @property
    def shape(self):
        return self._shape

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def v_range(self):
        return self._v_range

    @property
    def name(self):
        return self._name

    @property
    def rotatable(self):
        return self._rotatable

    @property
    def color(self):
        if isinstance(self._color, Color):
            return self._color.value
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def goal(self):
        return self._goal

    @property
    def drag(self):
        return self._drag

    @property
    def linear_friction(self):
        return self._linear_friction

    @linear_friction.setter
    def linear_friction(self, value):
        self._linear_friction = value

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        self._gravity = value

    @property
    def angular_friction(self):
        return self._angular_friction

    @goal.setter
    def goal(self, goal: Entity):
        self._goal = goal

    @property
    def collision_filter(self):
        return self._collision_filter

    @collision_filter.setter
    def collision_filter(self, collision_filter: Callable[[Entity], bool]):
        self._collision_filter = collision_filter

    def _spawn(self, dim_c: int, dim_p: int):
        self.state._spawn(dim_c, dim_p)

    def _reset(self, env_index: int):
        self.state._reset(env_index)

    def zero_grad(self):
        self.state.zero_grad()

    def set_pos(self, pos: Tensor, batch_index: int):
        self._set_state_property(EntityState.pos, self.state, pos, batch_index)

    def set_vel(self, vel: Tensor, batch_index: int):
        self._set_state_property(EntityState.vel, self.state, vel, batch_index)

    def set_rot(self, rot: Tensor, batch_index: int):
        self._set_state_property(EntityState.rot, self.state, rot, batch_index)

    def set_ang_vel(self, ang_vel: Tensor, batch_index: int):
        self._set_state_property(EntityState.ang_vel, self.state, ang_vel, batch_index)

    def _set_state_property(self, prop, entity: EntityState, new: Tensor, batch_index: int):
        assert self.batch_dim is not None, f'Tried to set property of {self.name} without adding it to the world'
        self._check_batch_index(batch_index)
        new = new.to(self.device)
        if batch_index is None:
            if len(new.shape) > 1 and new.shape[0] == self.batch_dim:
                prop.fset(entity, new)
            else:
                prop.fset(entity, new.repeat(self.batch_dim, 1))
        else:
            value = prop.fget(entity)
            value[batch_index] = new
        self.notify_observers()

    @override(TorchVectorizedObject)
    def to(self, device: torch.device):
        super().to(device)
        self.state.to(device)

    def render(self, env_index: int=0) -> 'List[Geom]':
        from vmas.simulator import rendering
        if not self.is_rendering[env_index]:
            return []
        geom = self.shape.get_geometry()
        xform = rendering.Transform()
        geom.add_attr(xform)
        xform.set_translation(*self.state.pos[env_index])
        xform.set_rotation(self.state.rot[env_index])
        color = self.color
        if isinstance(color, torch.Tensor) and len(color.shape) > 1:
            color = color[env_index]
        geom.set_color(*color)
        return [geom]


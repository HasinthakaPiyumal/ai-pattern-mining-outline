# Cluster 13

class EnvSpec(object):
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        trials (int): The number of trials to average reward over
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        local_only: True iff the environment is to be used only on the local machine (e.g. debugging envs)
        kwargs (dict): The kwargs to pass to the environment class
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including simple property=True tags

    Attributes:
        id (str): The official environment ID
        trials (int): The number of trials run in official evaluation
    """

    def __init__(self, id, entry_point=None, trials=100, reward_threshold=None, local_only=False, kwargs=None, nondeterministic=False, tags=None, max_episode_steps=None, max_episode_seconds=None, timestep_limit=None):
        self.id = id
        self.trials = trials
        self.reward_threshold = reward_threshold
        self.nondeterministic = nondeterministic
        if tags is None:
            tags = {}
        self.tags = tags
        if tags.get('wrapper_config.TimeLimit.max_episode_steps'):
            max_episode_steps = tags.get('wrapper_config.TimeLimit.max_episode_steps')
        tags['wrapper_config.TimeLimit.max_episode_steps'] = max_episode_steps
        if timestep_limit is not None:
            max_episode_steps = timestep_limit
        self.max_episode_steps = max_episode_steps
        self.max_episode_seconds = max_episode_seconds
        match = env_id_re.search(id)
        if not match:
            raise error.Error('Attempted to register malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id, env_id_re.pattern))
        self._env_name = match.group(1)
        self._entry_point = entry_point
        self._local_only = local_only
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self._entry_point is None:
            raise error.Error('Attempting to make deprecated env {}. (HINT: is there a newer registered version of this env?)'.format(self.id))
        elif callable(self._entry_point):
            env = self._entry_point(**kwargs)
        else:
            cls = load(self._entry_point)
            env = cls(**self._kwargs, **kwargs)
        env.unwrapped.spec = self
        return env

    def __repr__(self):
        return 'EnvSpec({})'.format(self.id)

    @property
    def timestep_limit(self):
        return self.max_episode_steps

    @timestep_limit.setter
    def timestep_limit(self, value):
        self.max_episode_steps = value

def load(name):
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    return result

def wrap_environment(wrapped_class, wrappers=None, **kwargs):
    """Helper for wrapping environment classes."""
    if wrappers is None:
        wrappers = []
    env_class = load(wrapped_class)
    env = env_class(**kwargs)
    for wrapper, wrapper_kwargs in wrappers:
        wrapper_class = load(wrapper)
        wrapper = wrapper_class(**wrapper_kwargs)
        env = wrapper(env)
    return env

def wrapper(env, config):
    scenario = os.path.join(ASSET_PATH, config['scenario'])
    map_name = config.get('map', 'MAP01').upper()
    cache_key = (scenario, map_name)
    if cache_key not in _MAP_CACHE:
        wad = omg.WadIO(scenario)
        editor = omg.UDMFMapEditor(wad)
        editor.load(map_name)
        _MAP_CACHE[cache_key] = editor
    else:
        editor = _MAP_CACHE[cache_key]
    editor = copy.deepcopy(editor)
    sampler(env, config, editor)
    updated_wad = tempfile.mktemp(suffix='.wad')
    editor.save(updated_wad)
    return updated_wad

def mujoco_wrapper(entry_point, **kwargs):
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    return env


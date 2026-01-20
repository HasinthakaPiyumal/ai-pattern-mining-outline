# Cluster 67

class FactorioGymRegistry:
    """Registry for Factorio gym environments"""

    def __init__(self):
        self._environments: Dict[str, GymEnvironmentSpec] = {}
        self._discovered = False

    def discover_tasks(self) -> None:
        """Automatically discover all task definitions and register them as gym environments"""
        if self._discovered:
            return
        from fle.eval.tasks.task_definitions.task_registry import list_all_tasks, get_task_info
        for task_key in list_all_tasks():
            task_info = get_task_info(task_key)
            self.register_environment(task_key=task_key, task_config_path=task_key, description=task_info['goal_description'], num_agents=task_info['num_agents'])
        self._discovered = True

    def register_environment(self, task_key: str, task_config_path: str, description: str, num_agents: int, enable_vision: bool=False) -> None:
        """Register a new gym environment"""
        spec = GymEnvironmentSpec(task_key=task_key, task_config_path=task_config_path, description=description, num_agents=num_agents, enable_vision=enable_vision)
        self._environments[task_key] = spec
        gym.register(id=task_key, entry_point='fle.env.gym_env.registry:make_factorio_env', kwargs={'spec': spec})

    def list_environments(self) -> List[str]:
        """List all registered environment IDs"""
        return list(self._environments.keys())

    def get_environment_spec(self, env_id: str) -> Optional[GymEnvironmentSpec]:
        """Get environment specification by ID"""
        return self._environments.get(env_id)

def list_all_tasks() -> list[str]:
    """Get a list of all available task keys."""
    return _task_registry.list_all_tasks()

def get_task_info(task_key: str) -> Dict[str, Any]:
    """Get detailed information about a task."""
    return _task_registry.get_task_info(task_key)


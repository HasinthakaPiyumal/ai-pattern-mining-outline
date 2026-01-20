# Cluster 85

class TaskRegistry:
    """Central registry for all task definitions."""

    def __init__(self):
        """Initialize the task registry with all available tasks."""
        self._all_tasks = {}
        self._all_tasks.update(THROUGHPUT_TASKS)
        self._all_tasks.update(UNBOUNDED_TASKS)
        self._all_tasks.update(MULTIAGENT_TASKS)
        self._task_type_to_class = {'throughput': ThroughputTask, 'unbounded_throughput': UnboundedThroughputTask, 'default': DefaultTask}

    def get_task_config(self, task_key: str) -> BaseModel:
        """Get a task configuration by its key.

        Args:
            task_key: The task identifier

        Returns:
            Task configuration instance

        Raises:
            KeyError: If the task_key doesn't exist
        """
        if task_key not in self._all_tasks:
            available = list(self._all_tasks.keys())
            raise KeyError(f'Unknown task: {task_key}. Available tasks: {', '.join(available[:5])}... ({len(available)} total)')
        return self._all_tasks[task_key]

    def create_task(self, task_key: str) -> TaskABC:
        """Create a task instance from its key.

        Args:
            task_key: The task identifier

        Returns:
            TaskABC instance ready to use

        Raises:
            KeyError: If the task_key doesn't exist
            ValueError: If the task type is not supported
        """
        task_config = self.get_task_config(task_key)
        config_dict = task_config.to_dict()
        task_type = config_dict.pop('task_type')
        config_dict.pop('num_agents', None)
        if task_type not in self._task_type_to_class:
            raise ValueError(f'Unsupported task type: {task_type}')
        task_class = self._task_type_to_class[task_type]
        return task_class(**config_dict)

    def list_all_tasks(self) -> list[str]:
        """Get a list of all available task keys."""
        return list(self._all_tasks.keys())

    def list_tasks_by_category(self) -> Dict[str, list[str]]:
        """Get tasks organized by category."""
        return {'throughput': list_throughput_tasks(), 'unbounded': list_unbounded_tasks(), 'multiagent': list_multiagent_tasks()}

    def get_task_info(self, task_key: str) -> Dict[str, Any]:
        """Get detailed information about a task.

        Args:
            task_key: The task identifier

        Returns:
            Dictionary with task information including type, description, etc.
        """
        config = self.get_task_config(task_key)
        config_dict = config.to_dict()
        return {'task_key': task_key, 'task_type': config_dict.get('task_type'), 'num_agents': config_dict.get('num_agents', 1), 'goal_description': config_dict.get('goal_description'), 'trajectory_length': config_dict.get('trajectory_length')}

    def task_exists(self, task_key: str) -> bool:
        """Check if a task exists in the registry.

        Args:
            task_key: The task identifier

        Returns:
            True if the task exists, False otherwise
        """
        return task_key in self._all_tasks

def list_throughput_tasks() -> list[str]:
    """Get a list of all available throughput task keys."""
    return list(THROUGHPUT_TASKS.keys())

def list_unbounded_tasks() -> list[str]:
    """Get a list of all available unbounded task keys."""
    return list(UNBOUNDED_TASKS.keys())

def list_multiagent_tasks() -> list[str]:
    """Get a list of all available multiagent task keys."""
    return list(MULTIAGENT_TASKS.keys())


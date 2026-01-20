# Cluster 48

def wrapper(self, *args, **kwargs):
    if self.random_state is None:
        return function(self, *args, **kwargs)
    else:
        with set_random_state(self.random_state, self.set_random_state):
            return function(self, *args, **kwargs)

@contextlib.contextmanager
def set_random_state(random_state, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_state (int or np.random.RandomState):
            The random seed or RandomState.
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_state = np.random.get_state()
    np.random.set_state(random_state.get_state())
    try:
        yield
    finally:
        current_random_state = np.random.RandomState()
        current_random_state.set_state(np.random.get_state())
        set_model_random_state(current_random_state)
        np.random.set_state(original_state)


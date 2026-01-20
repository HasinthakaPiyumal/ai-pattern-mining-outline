# Cluster 45

def sample_univariate_bimodal(size=1000, seed=42):
    """Sample from a bimodal distribution which mixes two Gaussians at 0.0 and 10.0 with stdev=1.

    The distribution is built by sampling a standard normal and a normal with mean ``10``
    and then selecting one or the other based on a bernoulli distribution.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        bernoulli = sample_univariate_bernoulli(size, seed)
        mode1 = np.random.normal(size=size) * bernoulli
        mode2 = np.random.normal(size=size, loc=10) * (1.0 - bernoulli)
        return pd.Series(mode1 + mode2)

def sample_univariate_bernoulli(size=1000, seed=42):
    """Sample from a Bernoulli distribution with p=0.3.

    The distribution is built by sampling a uniform random and then setting
    0 or 1 depending on whether the value is above or below 0.3.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(np.random.random(size=size) < 0.3).astype(float)

def sample_univariates(size=1000, seed=42):
    """Sample from a list of univariate distributions.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.DataFrame:
            DataFrame with the sampled distributions.
    """
    return pd.DataFrame({'bernoulli': sample_univariate_bernoulli(size, seed), 'bimodal': sample_univariate_bimodal(size, seed), 'uniform': sample_univariate_uniform(size, seed), 'normal': sample_univariate_normal(size, seed), 'degenerate': sample_univariate_degenerate(size, seed), 'exponential': sample_univariate_exponential(size, seed), 'beta': sample_univariate_beta(size, seed)})

def sample_univariate_uniform(size=1000, seed=42):
    """Sample from a uniform distribution in [-1.0, 3.0].

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(4.0 * np.random.random(size=size) - 1.0)

def sample_univariate_normal(size=1000, seed=42):
    """Sample from a normal distribution with mean 1 and stdev 1.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(np.random.normal(size=size, loc=1.0))

def sample_univariate_degenerate(size=1000, seed=42):
    """Sample from a degenerate distribution that only takes one random value.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(np.full(size, np.random.random()))

def sample_univariate_exponential(size=1000, seed=42):
    """Sample from an exponential distribution at 3.0 with rate 1.0.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(np.random.exponential(size=size) + 3.0)

def sample_univariate_beta(size=1000, seed=42):
    """Sample from a beta distribution with a=3 and b=1 and loc=4.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(stats.beta.rvs(a=3, b=1, loc=4, size=size))


# Cluster 13

def enable_data_source_cache(namespace: str, cache_dir: Optional[str]=None) -> Cache:
    """Enables caching of data retrieved from
    :class:`pybroker.data.DataSource`\\ s.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached data.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, 'data_source')
    scope.data_source_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.data_source_cache = cache
    scope.logger.debug_enable_data_source_cache(namespace, cache_dir)
    return cache

def _get_cache_dir(cache_dir: Optional[str], namespace: str, sub_dir: str) -> str:
    if not namespace:
        raise ValueError('Cache namespace cannot be empty.')
    base_dir = os.path.join(os.getcwd(), _DEFAULT_CACHE_DIRNAME) if cache_dir is None else cache_dir
    return os.path.join(base_dir, namespace, sub_dir)

def enable_indicator_cache(namespace: str, cache_dir: Optional[str]=None) -> Cache:
    """Enables caching indicator data.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached indicator data.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, 'indicator')
    scope.indicator_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.indicator_cache = cache
    scope.logger.debug_enable_indicator_cache(namespace, cache_dir)
    return cache

def enable_model_cache(namespace: str, cache_dir: Optional[str]=None) -> Cache:
    """Enables caching trained models.

    Args:
        namespace: Namespace of the cache.
        cache_dir: Directory used to store cached models.

    Returns:
        :class:`diskcache.Cache` instance.
    """
    scope = StaticScope.instance()
    cache_dir = _get_cache_dir(cache_dir, namespace, 'model')
    scope.model_cache_ns = namespace
    cache = Cache(directory=cache_dir)
    scope.model_cache = cache
    scope.logger.debug_enable_model_cache(namespace, cache_dir)
    return cache

def enable_caches(namespace, cache_dir: Optional[str]=None):
    """Enables all caches.

    Args:
        namespace: Namespace shared by cached data.
        cache_dir: Directory used to store cached data.
    """
    enable_data_source_cache(namespace, cache_dir)
    enable_indicator_cache(namespace, cache_dir)
    enable_model_cache(namespace, cache_dir)

def disable_caches():
    """Disables all caches."""
    disable_data_source_cache()
    disable_indicator_cache()
    disable_model_cache()

def disable_data_source_cache():
    """Disables caching data retrieved from
    :class:`pybroker.data.DataSource`\\ s.
    """
    scope = StaticScope.instance()
    scope.data_source_cache = None
    scope.data_source_cache_ns = ''
    scope.logger.debug_disable_data_source_cache()

def disable_indicator_cache():
    """Disables caching indicator data."""
    scope = StaticScope.instance()
    scope.indicator_cache = None
    scope.indicator_cache_ns = ''
    scope.logger.debug_disable_indicator_cache()

def disable_model_cache():
    """Disables caching trained models."""
    scope = StaticScope.instance()
    scope.model_cache = None
    scope.model_cache_ns = ''
    scope.logger.debug_disable_model_cache()

def clear_caches():
    """Clears cached data from all caches. :meth:`enable_caches` must be
    called first before clearing."""
    clear_data_source_cache()
    clear_indicator_cache()
    clear_model_cache()

def clear_data_source_cache():
    """Clears data cached from :class:`pybroker.data.DataSource`\\ s.
    :meth:`enable_data_source_cache` must be called first before clearing.
    """
    scope = StaticScope.instance()
    cache = scope.data_source_cache
    if cache is None:
        raise ValueError('Data source cache needs to be enabled before clearing.')
    cache.clear()
    scope.logger.debug_clear_data_source_cache(cache.directory)

def clear_indicator_cache():
    """Clears cached indicator data. :meth:`enable_indicator_cache` must be
    called first before clearing.
    """
    scope = StaticScope.instance()
    cache = scope.indicator_cache
    if cache is None:
        raise ValueError('Indicator cache needs to be enabled before clearing.')
    cache.clear()
    scope.logger.debug_clear_indicator_cache(cache.directory)

def clear_model_cache():
    """Clears cached trained models. :meth:`enable_model_cache` must be called
    first before clearing.
    """
    scope = StaticScope.instance()
    cache = scope.model_cache
    if cache is None:
        raise ValueError('Model cache needs to be enabled before clearing.')
    cache.clear()
    scope.logger.debug_clear_model_cache(cache.directory)

@pytest.fixture()
def setup_enabled_model_cache(tmp_path):
    enable_model_cache('test', tmp_path)
    yield
    clear_model_cache()
    disable_model_cache()

@pytest.fixture(params=[True, False])
def setup_model_cache(tmp_path, request):
    if request.param:
        enable_model_cache('test', tmp_path)
    else:
        disable_model_cache()
    yield
    if request.param:
        clear_model_cache()
    disable_model_cache()

@pytest.fixture()
def setup_enabled_ds_cache(tmp_path):
    enable_data_source_cache('test', tmp_path)
    yield
    clear_data_source_cache()
    disable_data_source_cache()

@pytest.fixture(params=[True, False])
def setup_ds_cache(tmp_path, request):
    if request.param:
        enable_data_source_cache('test', tmp_path)
    else:
        disable_data_source_cache()
    yield
    if request.param:
        clear_data_source_cache()
    disable_data_source_cache()

@pytest.fixture()
def setup_enabled_ind_cache(tmp_path):
    enable_indicator_cache('test', tmp_path)
    yield
    clear_indicator_cache()
    disable_indicator_cache()

@pytest.fixture(params=[True, False])
def setup_ind_cache(tmp_path, request):
    if request.param:
        enable_indicator_cache('test', tmp_path)
    else:
        disable_indicator_cache()
    yield
    if request.param:
        clear_indicator_cache()
    disable_indicator_cache()

@pytest.mark.usefixtures('setup_teardown')
def test_enable_and_disable_all_caches(scope, cache_dir, cache_path):
    enable_caches('test', cache_dir)
    assert len(list(cache_path.iterdir())) == 1
    assert isinstance(scope.data_source_cache, Cache)
    assert isinstance(scope.indicator_cache, Cache)
    assert isinstance(scope.model_cache, Cache)
    assert scope.data_source_cache_ns == 'test'
    assert scope.indicator_cache_ns == 'test'
    assert scope.model_cache_ns == 'test'
    disable_caches()
    assert scope.data_source_cache is None
    assert scope.indicator_cache is None
    assert scope.model_cache is None
    assert scope.data_source_cache_ns == ''
    assert scope.indicator_cache_ns == ''
    assert scope.model_cache_ns == ''

@pytest.mark.usefixtures('setup_teardown')
def test_clear_all_caches(scope, cache_dir):
    enable_caches('test', cache_dir)
    with mock.patch.object(scope, 'data_source_cache') as data_source_cache, mock.patch.object(scope, 'indicator_cache') as ind_cache, mock.patch.object(scope, 'model_cache') as model_cache:
        clear_caches()
        data_source_cache.clear.assert_called_once()
        ind_cache.clear.assert_called_once()
        model_cache.clear.assert_called_once()


# Cluster 41

def evaluate_transformer_performance(transformer, dataset_generator, verbose=False):
    """Evaluate the given transformer's performance against the given dataset generator.

    Args:
        transformer (rdt.transformers.BaseTransformer):
            The transformer to evaluate.
        dataset_generator (rdt.tests.datasets.BaseDatasetGenerator):
            The dataset generator to performance test against.
        verbose (bool):
            Whether or not to add extra columns about the dataset and transformer,
            and return data for all dataset sizes. If false, it will only return
            the max performance values of all the dataset sizes used.

    Returns:
        pandas.DataFrame:
            The performance test results.
    """
    transformer_args = TRANSFORMER_ARGS.get(transformer.__name__, {})
    transformer_instance = transformer(**transformer_args)
    sizes = _get_dataset_sizes(dataset_generator.SDTYPE)
    out = []
    for fit_size, transform_size in sizes:
        performance = profile_transformer(transformer=transformer_instance, dataset_generator=dataset_generator, fit_size=fit_size, transform_size=transform_size)
        size = np.array([fit_size, transform_size, transform_size] * 2)
        performance = performance / size
        if verbose:
            performance = performance.rename(lambda x: x + ' (s)' if 'Time' in x else x + ' (B)')
            performance['Number of fit rows'] = fit_size
            performance['Number of transform rows'] = transform_size
            performance['Dataset'] = dataset_generator.__name__
            performance['Transformer'] = f'{transformer.__module__}.{transformer.__name__}'
        out.append(performance)
    summary = pd.DataFrame(out)
    if verbose:
        return summary
    return summary.max(axis=0)

def _get_dataset_sizes(sdtype):
    """Get a list of (fit_size, transform_size) for each dataset generator.

    Based on the sdtype of the dataset generator, return the list of
    sizes to run performance tests on. Each element in this list is a tuple
    of (fit_size, transform_size).

    Args:
        sdtype (str):
            The type of data that the generator returns.

    Returns:
        sizes (list[tuple]):
            A list of (fit_size, transform_size) configs to run tests on.
    """
    sizes = [(s, s) for s in DATASET_SIZES]
    if sdtype == 'categorical':
        sizes = [(s, max(s, 1000)) for s in DATASET_SIZES if s <= 10000]
    return sizes

def profile_transformer(transformer, dataset_generator, transform_size, fit_size=None):
    """Profile a Transformer on a dataset.

    This function will get the total time and peak memory
    for the ``fit``, ``transform`` and ``reverse_transform``
    methods of the provided transformer against the provided
    dataset.

    Args:
        transformer (Transformer):
            Transformer instance.
        dataset_generator (DatasetGenerator):
            DatasetGenerator instance.
        transform_size (int):
            Number of rows to generate for ``transform`` and ``reverse_transform``.
        fit_size (int or None):
            Number of rows to generate for ``fit``. If None, use ``transform_size``.

    Returns:
        pandas.Series:
            Series containing the time and memory taken by ``fit``, ``transform``,
            and ``reverse_transform`` for the transformer.
    """
    fit_size = fit_size or transform_size
    fit_dataset = pd.DataFrame({'test': dataset_generator.generate(fit_size)})
    replace = transform_size > fit_size
    transform_dataset = fit_dataset.sample(transform_size, replace=replace)
    fit_time = _profile_time(transformer, 'fit', fit_dataset, column='test', copy=True)
    fit_memory = _profile_memory(transformer.fit, fit_dataset, column='test')
    transformer.fit(fit_dataset, 'test')
    transform_time = _profile_time(transformer, 'transform', transform_dataset)
    transform_memory = _profile_memory(transformer.transform, transform_dataset)
    reverse_dataset = transformer.transform(transform_dataset)
    reverse_time = _profile_time(transformer, 'reverse_transform', reverse_dataset)
    reverse_memory = _profile_memory(transformer.reverse_transform, reverse_dataset)
    return pd.Series({'Fit Time': fit_time, 'Fit Memory': fit_memory, 'Transform Time': transform_time, 'Transform Memory': transform_memory, 'Reverse Transform Time': reverse_time, 'Reverse Transform Memory': reverse_memory})

def _profile_time(transformer, method_name, dataset, column=None, iterations=10, copy=False):
    total_time = 0
    for _ in range(iterations):
        if copy:
            transformer_copy = deepcopy(transformer)
            method = getattr(transformer_copy, method_name)
        else:
            method = getattr(transformer, method_name)
        start_time = timeit.default_timer()
        if column:
            method(dataset, column)
        else:
            method(dataset)
        total_time += timeit.default_timer() - start_time
    return total_time / iterations

def _profile_memory(method, dataset, column=None):
    ctx = mp.get_context('spawn')
    peak_memory = ctx.Value('i', 0)
    profiling_process = ctx.Process(target=_set_memory_for_method, args=(method, dataset, column, peak_memory))
    profiling_process.start()
    profiling_process.join()
    return peak_memory.value


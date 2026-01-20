# Cluster 44

def get_dataset_generators_by_type():
    """Build a ``dict`` mapping sdtypes to dataset generators.

    Returns:
        dict:
            Mapping of sdtype to a list of dataset generators that produce
            data of that sdtype.
    """
    dataset_generators = defaultdict(list)
    for dataset_generator in BaseDatasetGenerator.get_subclasses():
        dataset_generators[dataset_generator.SDTYPE].append(dataset_generator)
    return dataset_generators


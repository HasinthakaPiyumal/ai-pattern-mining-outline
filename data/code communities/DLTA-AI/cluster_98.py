# Cluster 98

def _check_numclasscheckhook(detector, config_mod):
    dummy_runner = Mock()
    dummy_runner.model = detector

    def get_dataset_name_classes(dataset):
        if isinstance(dataset, (list, tuple)):
            dataset = dataset[0]
        while 'dataset' in dataset:
            dataset = dataset['dataset']
            if isinstance(dataset, (list, tuple)):
                dataset = dataset[0]
        return (dataset['type'], dataset.get('classes', None))
    compatible_check = NumClassCheckHook()
    dataset_name, CLASSES = get_dataset_name_classes(config_mod['data']['train'])
    if CLASSES is None:
        CLASSES = DATASETS.get(dataset_name).CLASSES
    dummy_runner.data_loader.dataset.CLASSES = CLASSES
    compatible_check.before_train_epoch(dummy_runner)
    dummy_runner.data_loader.dataset.CLASSES = None
    compatible_check.before_train_epoch(dummy_runner)
    dataset_name, CLASSES = get_dataset_name_classes(config_mod['data']['val'])
    if CLASSES is None:
        CLASSES = DATASETS.get(dataset_name).CLASSES
    dummy_runner.data_loader.dataset.CLASSES = CLASSES
    compatible_check.before_val_epoch(dummy_runner)
    dummy_runner.data_loader.dataset.CLASSES = None
    compatible_check.before_val_epoch(dummy_runner)

def get_dataset_name_classes(dataset):
    if isinstance(dataset, (list, tuple)):
        dataset = dataset[0]
    while 'dataset' in dataset:
        dataset = dataset['dataset']
        if isinstance(dataset, (list, tuple)):
            dataset = dataset[0]
    return (dataset['type'], dataset.get('classes', None))


# Cluster 24

def _set_splits_samples(split2samples: DefaultDict[str, List[Sample]], db: NuPlanDB, split2log: Dict[str, List[str]], broken_extractions: Optional[Set[str]]=None, sort_train: bool=True) -> None:
    """
    Populates split2samples with all the main splits and the ones defined in split2log, converting log names into
    the list of non-broken samples they contain in the database.

    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data.
        The data is given as a list of samples.
    :param db: NuPlanDB.
    :param split2log: Mapping from a split name to its corresponding data. The data is given as a list of log names
        (example of log name: '2019.07.03.03.29.58_mvp-veh-8')
    :param broken_extractions: List of extractions whose samples should be excluded from the samples populated in
        `split2samples`.
    :param sort_train: Whether or not to sort the train split with respect to sample tokens. (This is useful
        to guarantee that randomly subsampled splits from train will not differ if they have the same random seed.)
    """
    broken_extractions = broken_extractions or set()
    for split_name in split2log.keys():
        logs = _get_logs(db, split2log, split_name)
        split2samples[split_name] = _get_samples_from_logs(logs, broken_extractions=broken_extractions)
    if 'val' in split2samples or 'test' in split2samples:
        split2samples['valtest'] = split2samples['val'] + split2samples['test']
    if 'train' not in split2samples:
        split2samples['train'] = [rec for rec in split2samples['all'] if rec not in split2samples['valtest']]
    if sort_train:
        split2samples['train'].sort(key=lambda sample: str(sample.token))

def _get_logs(db: NuPlanDB, split2log: Dict[str, List[str]], split_name: str) -> List[Log]:
    """
    For all the given split `split_name`, convert its corresponding log names into Log objects.
    :param db: NuPlanDB.
    :param split2log: Mapping from a split name to its corresponding data. The data is given as a list of log names
        (example of log name: '2021.07.16.20.45.29_veh-35_01095_01486').
    :param split_name: The split in which we want to get the Log objects. (example of split_name: "val").
    :return: List of logs.
    """
    logs = []
    for log_name in split2log[split_name]:
        log = db.log.select_one(logfile=log_name)
        if log is not None:
            logs.append(log)
    return logs

def _get_samples_from_logs(logs: List[Log], broken_extractions: Set[str]) -> List[Sample]:
    """
    Returns all the non-broken samples of a list of logs.
    For definitions of 'sample' and 'extraction', please take a look at README.md.

    :param logs: List of logs from which to extract samples.
    :param broken_extractions: List of extractions whose samples should be excluded from the output of this function.
    :return: List of non-broken samples associated with the given logs.
    """
    samples = []
    for log in logs:
        for extraction in log.extractions:
            if extraction.token in broken_extractions:
                continue
            for sample in extraction.samples:
                samples.append(sample)
    return samples

def _get_all_samples(db: NuPlanDB, vehicle_type: str, broken_extractions: Optional[Set[str]]=None, excluded_drive_log_tags: Optional[Set[str]]=None) -> List[Sample]:
    """
    Returns all non-broken samples associated with one vehicle.

    :param db: NuPlanDB.
    :param vehicle_type: name of the vehicle which we should get data from.
    :param broken_extractions: List of extractions whose samples should be excluded from the output of this function.
    :param excluded_drive_log_tags: Logs that have ANY of those drive log tags will be excluded.
    :return: List of non-broken samples associated with given vehicle_type.
    """
    broken_extractions = broken_extractions or set()
    excluded_drive_log_tags = excluded_drive_log_tags or set()
    logs = db.log.select_many(vehicle_type=vehicle_type)
    logs = [log for log in logs if not set(log.drive_log_tags).intersection(excluded_drive_log_tags)]
    return _get_samples_from_logs(logs, broken_extractions)


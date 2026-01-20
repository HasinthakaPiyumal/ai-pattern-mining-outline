# Cluster 25

def _set_mini_splits(split2samples: DefaultDict[str, List[Sample]], db: NuPlanDB, core_splits_names: List[str]) -> None:
    """
    Populates split2samples with mini splits done on top of core splits.

    For example:
        "train" -> "train", "train.mini"

    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data. The data is
     given as a list of samples. This function assumes the existence the following splits:
      - core splits (e.g. "train", "val, "test").
      - location splits (e.g. "train.bs", "val.United_States").
    :param db: NuPlanDB.
    :param core_splits_names: Name of the core splits to be considered.
    """
    return _set_subsampled_splits(split2samples, db, core_splits_names, random_seed='42', n_samples_per_region=100, split_suffix='mini')

def _set_subsampled_splits(split2samples: DefaultDict[str, List[Sample]], db: NuPlanDB, core_splits_names: List[str], random_seed: Union[str, int], n_samples_per_region: int, split_suffix: str) -> None:
    """
    Populates split2samples with core splits.
    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data. The data is
     given as a list of samples.
    :param db: NuPlanDB.
    :param core_splits_names: Name of the core splits, such as, ['test', 'val', 'valtest', 'train'].
    :param random_seed: Random seed to use for picking tokens.
    :param n_samples_per_region: number of samples for each region.
    :param split_suffix: suffix of the split name, such as 'mini', 'dev'.
    """
    st0 = random.getstate()
    random.seed(random_seed)
    for split_name in core_splits_names:
        for region in db.regions:
            temp = split2samples[split_name + '.' + region].copy()
            random.shuffle(temp)
            split2samples[split_name + '.' + split_suffix] += temp[:n_samples_per_region]
    random.setstate(st0)

def _set_dev_splits(split2samples: DefaultDict[str, List[Sample]], db: NuPlanDB, core_splits_names: List[str]) -> None:
    """
    Populates split2samples with smaller evaluation splits done on top of core splits, to use in dev. experiments.
    For example:
        "train" -> "train", "train.dev"

    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data. The data is
     given as a list of samples. This function assumes the existence the following splits:
      - core splits (e.g. "train", "val, "test")
      - location splits (e.g. "train.bs", "val.United_States").
    :param db: NuPlanDB.
    :param core_splits_names: Name of the core splits to be considered.
    """
    return _set_subsampled_splits(split2samples, db, core_splits_names, random_seed='42', n_samples_per_region=250, split_suffix='dev')


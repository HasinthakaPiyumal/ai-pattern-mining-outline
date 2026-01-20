# Cluster 37

def load_code_generation_dataset(release_version='release_v1', cache_dir: str=None, start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    dataset = load_dataset('livecodebench/code_generation_lite', split='test', version_tag=release_version, trust_remote_code=True, cache_dir=cache_dir)
    dataset = [CodeGenerationProblem(**p) for p in dataset]
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, '%Y-%m-%d')
        dataset = [e for e in dataset if p_start_date <= e.contest_date]
    if end_date is not None:
        p_end_date = datetime.strptime(end_date, '%Y-%m-%d')
        dataset = [e for e in dataset if e.contest_date <= p_end_date]
    return dataset


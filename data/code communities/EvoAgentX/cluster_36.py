# Cluster 36

def load_code_execution_dataset(release_version='release_v1', cache_dir: str=None) -> list[CodeExecutionProblem]:
    dataset = load_dataset('livecodebench/execution-v2', split='test', trust_remote_code=True, cache_dir=cache_dir)
    dataset = [CodeExecutionProblem(**p) for p in dataset]
    return dataset


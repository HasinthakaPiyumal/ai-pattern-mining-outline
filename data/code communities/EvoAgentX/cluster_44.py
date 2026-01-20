# Cluster 44

def load_test_prediction_dataset(release_version='release_v1', cache_dir: str=None) -> list[TestOutputPredictionProblem]:
    dataset = load_dataset('livecodebench/test_generation', split='test', trust_remote_code=True, cache_dir=cache_dir)
    dataset = [TestOutputPredictionProblem(**d) for d in dataset]
    return dataset


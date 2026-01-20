# Cluster 30

class WorfBench(Benchmark):
    """
    WorfBench evaluation class for assessing LLM agents on complex workflow generation tasks.
    Assumed data structure:
    {
        "id": str,
        "task": str,
        "context": list of dicts (e.g., [{"title": str, "content": list of str}]),
        "expected_output": str or dict (sequence or graph),
        "type": str,
        "level": str
    }
    """

    def __init__(self, path: str=None, mode: str='test', **kwargs):
        path = os.path.expanduser(path or '~/.worfbench/data')
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str) -> Dict:
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_worfbench_data(dataset='worfbench', save_folder=self.path)
        if not os.path.exists(file_path):
            logger.error(f'File {file_path} still does not exist after download attempt!')
            return None
        logger.info(f'Loading WorfBench data from {file_path} ...')
        data = load_json(path=file_path, type='json')
        if data is None:
            logger.error(f'Failed to load data from {file_path}')
            return None
        return data

    def _load_data(self) -> None:
        if self.mode in ['train', 'dev']:
            self._train_data = self._load_data_from_file(file_name=WORFBENCH_FILES_MAP['train'])
            if self.mode == 'dev':
                if self._train_data:
                    random.seed(42)
                    keys = list(self._train_data.keys())
                    n_dev = len(self._train_data[keys[0]]) // 10 or 1
                    indices = list(range(len(self._train_data[keys[0]])))
                    random.shuffle(indices)
                    self._train_data = {k: [v[i] for i in indices[:n_dev]] for k, v in self._train_data.items()}
        if self.mode == 'test':
            self._test_data = self._load_data_from_file(file_name=WORFBENCH_FILES_MAP['test'])

    def _get_label(self, example: Dict) -> Any:
        return example.get('expected_output', '')

    def _get_id(self, example: Dict) -> Any:
        return example.get('id', '')

    def evaluate(self, prediction: Any, label: Any) -> Dict:
        if isinstance(prediction, list) and isinstance(label, list):
            f1 = evaluate_workflow_sequence(prediction, label)
        elif isinstance(prediction, dict) and isinstance(label, dict):
            f1 = evaluate_workflow_graph(prediction, label)
        else:
            f1 = f1_score(prediction=str(prediction), ground_truth=str(label))
        em = exact_match_score(prediction=prediction, ground_truth=label)
        acc = acc_score(prediction=prediction, ground_truths=[label])
        return {'em': em, 'f1': f1, 'acc': acc}

    async def async_evaluate(self, graph: Callable, example: Dict) -> float:
        task = example.get('task', '')
        context = '\n'.join((f'{ctx.get('title', '')}: {' '.join(ctx.get('content', []))}' for ctx in example.get('context', []) if isinstance(ctx, dict)))
        inputs = f'Task: {task}\nContext: {context}\nGenerate workflow:\nAnswer:'
        try:
            generated_workflow = await graph(inputs)
        except Exception as e:
            logger.error(f'Error generating workflow: {e}')
            generated_workflow = ''
        label = self._get_label(example)
        metrics = self.evaluate(prediction=generated_workflow, label=label)
        return metrics['f1']

def download_worfbench_data(dataset: str, save_folder: str) -> None:
    """
    Download WorfBench dataset from Hugging Face.

    Args:
        dataset (str): Dataset name ("worfbench").
        save_folder (str): Directory to save data.
    """
    datasets_map = {'train': {'repo_id': 'zjunlp/WorFBench_train', 'filename': 'worfbench_train.json', 'split': 'train'}, 'test': {'repo_id': 'zjunlp/WorFBench_test', 'filename': 'worfbench_test.json', 'split': 'test'}}
    os.makedirs(save_folder, exist_ok=True)
    for split, info in datasets_map.items():
        repo_id = info['repo_id']
        filename = info['filename']
        dataset_split = info['split']
        save_path = os.path.join(save_folder, filename)
        if not os.path.exists(save_path):
            logger.info(f'Downloading {split} split of {dataset} from {repo_id}...')
            try:
                ds = load_dataset(repo_id, split=dataset_split)
                data = [item for item in ds]
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f'Successfully downloaded and saved {filename} to {save_path}')
            except Exception as e:
                logger.error(f'Failed to download or save {filename}: {e}')
                raise
        else:
            logger.info(f'File {save_path} already exists, skipping download.')


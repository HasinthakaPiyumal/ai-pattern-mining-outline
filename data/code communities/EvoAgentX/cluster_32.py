# Cluster 32

class BIGBenchHard(Benchmark):
    """
    Benchmark class for BIGBenchHard dataset evaluation.
    
    BIGBenchHard is a subset of 23 challenging tasks from the BIG-bench evaluation suite.
    Each task example has the following structure:
    {
        "input": str,    # The input question/problem
        "target": str    # The expected answer/output
    }
    
    The benchmark supports automatic data splitting for training/validation purposes
    and evaluates predictions using exact match scoring.
    """

    def __init__(self, task: str, path: str=None, mode: str='all', dev_sample_num: int=0, seed: int=10, **kwargs):
        """
        Initialize BIGBenchHard benchmark.
        
        Args:
            task: The specific BIGBenchHard task name
            path: Path to store the dataset. Defaults to ~/.evoagentx/data/bigbenchhard/{task}
            mode: Data loading mode. Defaults to "all"
            dev_sample_num: Number of samples to use for dev set. If 0, all data goes to test set
            seed: Random seed for reproducibility. Defaults to 10
            **kwargs: Additional parameters for customization
            
        Raises:
            ValueError: If task is not a valid BIGBenchHard task name
        """
        if task not in ALL_TASKS:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {list(ALL_TASKS.keys())}")
        self.task = task
        self.file_name = ALL_TASKS[task]
        self.dev_sample_num = dev_sample_num
        self.seed = seed
        path = os.path.expanduser(path or f'~/.evoagentx/data/bigbenchhard/{task}')
        super().__init__(name=f'BIGBenchHard-{self.task}', path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str) -> Optional[List[dict]]:
        """
        Load data from a specific file.
        
        Args:
            file_name: Name of the file to load
            
        Returns:
            List of loaded examples or None if file doesn't exist
        """
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_bigbenchhard_data(task_name=self.task, save_folder=self.path)
        logger.info(f'Loading BIGBenchHard data from {file_path}...')
        data = load_json(path=file_path, type='json')
        return data.get('examples', [])

    def _load_data(self):
        """
        Load and split data according to mode and dev_sample_num settings.
        
        Data splitting logic:
        - If dev_sample_num > 0: randomly samples examples for dev set, rest go to test set
        - If dev_sample_num = 0: all data goes to test set for evaluation
        - No training data provided (BIGBenchHard is designed for few-shot evaluation)
        """
        task_data = self._load_data_from_file(file_name=self.file_name)
        if task_data is None:
            logger.warning(f'No data loaded for task {self.task}')
            self._train_data = []
            self._dev_data = []
            self._test_data = []
            return
        self._train_data = []
        if self.dev_sample_num > 0 and len(task_data) > self.dev_sample_num:
            logger.info(f'Sampling {self.dev_sample_num} examples for dev set, rest for test set.')
            if self.seed is not None:
                set_seed(self.seed)
            dev_subset = random.sample(task_data, self.dev_sample_num)
            self._dev_data = dev_subset
            self._test_data = [item for item in task_data if item not in dev_subset]
        elif self.dev_sample_num > 0:
            logger.warning(f'dev_sample_num ({self.dev_sample_num}) >= total data size ({len(task_data)}). Using all data for dev set, none for test set.')
            self._dev_data = task_data
            self._test_data = []
        else:
            logger.info('dev_sample_num is 0, using all data for test set.')
            self._dev_data = []
            self._test_data = task_data

    def get_input_keys(self) -> List[str]:
        """
        Return the input keys expected by the benchmark.
        
        Returns:
            List containing "input" as the key for the problem text
        """
        return ['input']

    def _get_label(self, example: Any) -> Any:
        """
        Extract the ground truth label from an example.
        
        Args:
            example: The benchmark example
            
        Returns:
            The target answer/label
        """
        return example['target']

    def _get_id(self, example: Any) -> Any:
        """
        Extract the unique identifier from an example.
        
        BIGBenchHard examples don't have explicit IDs, so we use input text as identifier.
        
        Args:
            example: The benchmark example
            
        Returns:
            The input text as a unique identifier
        """
        return example.get('input', None)

    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Score a prediction against the ground truth label.
        
        Uses exact match scoring with task-specific handling for certain tasks.
        
        Args:
            prediction: The predicted answer
            label: The ground truth answer
            
        Returns:
            Dictionary containing the exact match score
        """
        if self.task == 'dyck_languages':
            em = prediction.replace(' ', '') == label.replace(' ', '')
            return {'em': em}
        else:
            em = exact_match_score(prediction=prediction, ground_truth=label)
            return {'em': em}

def set_seed(seed: int):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed: The random seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


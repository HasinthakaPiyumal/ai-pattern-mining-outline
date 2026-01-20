# Cluster 10

def download_file(url: str, save_file: str, max_retries=3, timeout=10):
    make_parent_folder(save_file)
    for attempt in range(max_retries):
        try:
            resume_byte_pos = 0
            if os.path.exists(save_file):
                resume_byte_pos = os.path.getsize(save_file)
            response_head = requests.head(url=url)
            total_size = int(response_head.headers.get('content-length', 0))
            if resume_byte_pos >= total_size:
                logger.info('File already downloaded completely.')
                return
            headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos else {}
            response = requests.get(url=url, stream=True, headers=headers, timeout=timeout)
            response.raise_for_status()
            mode = 'ab' if resume_byte_pos else 'wb'
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, initial=resume_byte_pos)
            with open(save_file, mode) as file:
                for chunk_data in response.iter_content(chunk_size=1024):
                    if chunk_data:
                        size = file.write(chunk_data)
                        progress_bar.update(size)
            progress_bar.close()
            if os.path.getsize(save_file) >= total_size + resume_byte_pos:
                logger.info('Download completed successfully.')
                break
            else:
                logger.warning('File size mismatch, retrying...')
                time.sleep(5)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f'Download error: {e}. Retrying ({attempt + 1}/{max_retries})...')
            time.sleep(5)
        except Exception as e:
            error_message = f'Unexpected error: {e}'
            logger.error(error_message)
            raise ValueError(error_message)
    else:
        error_message = 'Exceeded maximum retries. Download failed.'
        logger.error(error_message)
        raise RuntimeError(error_message)

def download_raw_bigbenchhard_data(task_name: str, save_folder: str):
    """
    Download raw BIGBenchHard data for a specific task.
    
    Args:
        task_name: The name of the task to download
        save_folder: Directory to save the downloaded data file
        
    Raises:
        AssertionError: If task_name is not a valid BIGBenchHard task
    """
    assert task_name in ALL_TASKS, f"'{task_name}' is an invalid bigbenchhard task name. Available tasks: {list(ALL_TASKS.keys())}"
    file_name = ALL_TASKS[task_name]
    url = f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{file_name}'
    logger.info(f"Downloading BIGBenchHard '{task_name}' data from: {url}")
    download_file(url=url, save_file=os.path.join(save_folder, file_name))

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

def download_raw_mbpp_data(name: str, save_folder: str):
    url = 'https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json'
    logger.info(f'Downloading MBPP data from: {url}')
    download_file(url=url, save_file=os.path.join(save_folder, name))

def extract_func_header(code: str, test_list: List[str]) -> str:
    lines = code.split('\n')
    imports, defs = ([], [])
    for line in lines:
        if line.startswith('def '):
            break
        imports.append(line)
    for line in lines:
        if line.startswith('def '):
            defs.append(line)
    func_head = None
    for header in defs:
        func_name = extract_func_name(header)
        if func_name is None:
            continue
        if all((func_name in test for test in test_list)):
            func_head = header
            break
    if func_head is None:
        logger.warning(f'No function header found for {code}')
    return ('\n'.join(imports) + '\n\n' + func_head).strip()

def extract_func_name(func_header: str) -> str:
    func_name_pattern = 'def\\s+([a-zA-Z_]\\w*)\\s*\\('
    match = regex.search(func_name_pattern, func_header)
    if match:
        return match.group(1)
    else:
        return None

def load_mbpp_data(data_path: str):
    """
    load MBPP data from the given path and convert to HumanEval format
    """

    def extract_func_name(func_header: str) -> str:
        func_name_pattern = 'def\\s+([a-zA-Z_]\\w*)\\s*\\('
        match = regex.search(func_name_pattern, func_header)
        if match:
            return match.group(1)
        else:
            return None

    def extract_func_header(code: str, test_list: List[str]) -> str:
        lines = code.split('\n')
        imports, defs = ([], [])
        for line in lines:
            if line.startswith('def '):
                break
            imports.append(line)
        for line in lines:
            if line.startswith('def '):
                defs.append(line)
        func_head = None
        for header in defs:
            func_name = extract_func_name(header)
            if func_name is None:
                continue
            if all((func_name in test for test in test_list)):
                func_head = header
                break
        if func_head is None:
            logger.warning(f'No function header found for {code}')
        return ('\n'.join(imports) + '\n\n' + func_head).strip()
    data = load_json(data_path, type='json')
    for example in data:
        original_prompt = example['prompt']
        code = example['code']
        test_list = [assert_str.strip() for assert_str in example['test_list']]
        func_header = extract_func_header(code, test_list)
        if example['task_id'] == 56:
            func_header = func_header.replace('check', 'check_answer')
            code = code.replace('check', 'check_answer')
            test_list = [test.replace('check', 'check_answer') for test in test_list]
        prompt = example['prompt'] + '\n\n' + func_header + '\n'
        canonical_solution = code
        test = 'def check(candidate):\n    ' + '\n    '.join(test_list) + '\n'
        entry_point = extract_func_name(func_header)
        example['prompt'] = prompt
        example['entry_point'] = entry_point
        example['canonical_solution'] = canonical_solution
        example['test'] = test
        example['original_prompt'] = original_prompt
    return data

class MBPP(CodingBenchmark):
    """Benchmark class for evaluating code generation on the MBPP dataset.
    
    MBPP (Mostly Basic Python Programming) is a collection of Python programming 
    problems designed to test a model's ability to generate functionally correct 
    code from natural language descriptions. This class handles loading the dataset, 
    evaluating solutions, and computing metrics such as pass@k.
    
    The original MBPP format is transformed to be compatible with the HumanEval
    benchmark format, allowing for consistent evaluation infrastructure.
    
    Each MBPP example has the following structure:
    {
        "task_id" (int): 2, 
        "prompt" (str): "Write a function to find the shared elements from the given two lists.",
        "code" (str): "def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res) ", 
        "test_imports": [] 
        "test_list" (List[str]): ['assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))', 'assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))', 'assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))']
    }
    
    Attributes:
        k: An integer or list of integers specifying which pass@k metrics to compute
    """

    def __init__(self, path: str=None, mode: str='all', timeout: int=60, k: Union[int, list]=1, **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/mbpp')
        self.k = k
        super().__init__(name=type(self).__name__, path=path, mode=mode, timeout=timeout, **kwargs)

    def _load_data(self):
        data_path = os.path.join(self.path, 'sanitized-mbpp.json')
        if not os.path.exists(data_path):
            download_raw_mbpp_data(name='sanitized-mbpp.json', save_folder=self.path)
        if self.mode == 'train' or self.mode == 'all':
            self._train_data = None
        if self.mode == 'dev' or self.mode == 'all':
            self._dev_data = None
        if self.mode == 'test' or self.mode == 'all':
            self._test_data = load_mbpp_data(data_path)

    def _get_id(self, example: Any) -> Any:
        return example['task_id']

    def _get_label(self, example: Any) -> Any:
        return {'task_id': example['task_id'], 'canonical_solution': example['canonical_solution'], 'test': example['test'], 'entry_point': example['entry_point']}

    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Evaluate the solution code.

        Args:
            prediction (str | List[str]): The solution code(s).
            label (dict | List[dict]): The unit test code(s).

        Returns:
            dict: The evaluation metrics (pass@k).
        """
        prediction, label = self._check_evaluation_inputs(prediction, label)
        results = []
        for solution in prediction:
            solution_states = []
            for label_data in label:
                task_id = label_data['task_id']
                prompt = self.get_example_by_id(task_id)['prompt']
                unit_test = label_data['test']
                entry_point = label_data['entry_point']
                state, message = self.check_solution(task_id=task_id, solution=prompt + '\n' + solution, test=unit_test, entry_point=entry_point)
                if state != self.SUCCESS:
                    break
                solution_states.append(state)
            results.append(len(solution_states) == len(label) and all((state == self.SUCCESS for state in solution_states)))
        k_list = [self.k] if isinstance(self.k, int) else self.k
        pass_at_k = self.compute_pass_at_k(results, k_list)
        return pass_at_k

def download_raw_humaneval_data(save_folder: str):
    url = 'https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz'
    logger.info(f'Downloading HumanEval data from {url} ...')
    save_file_path = os.path.join(save_folder, 'HumanEval.jsonl.gz')
    download_file(url=url, save_file=save_file_path)
    with gzip.open(save_file_path, 'rb') as f_in, open(os.path.join(save_folder, 'HumanEval.jsonl'), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    if os.path.exists(save_file_path):
        os.remove(save_file_path)

class HumanEval(CodingBenchmark):
    """Benchmark class for evaluating code generation on HumanEval.
    
    HumanEval is a collection of Python programming problems designed to test
    a model's ability to generate functionally correct code from natural language
    descriptions. This class handles loading the dataset, evaluating solutions,
    and computing metrics such as pass@k.
    
    Each HumanEval example has the following structure:
    {
        "task_id": "HumanEval/0", 
        "prompt": "from typing import List

def func_name(*args, **kwargs) -> return_type
    "function description"

", 
        "entry_point": "func_name",
        "canonical_solution": "canonical solution (code)",
        "test": "METADATA = {xxx}


def check(candidate):
 assert candidate(inputs) == output
"
    }
    
    Attributes:
        k: An integer or list of integers specifying which pass@k metrics to compute
    """

    def __init__(self, path: str=None, mode: str='all', timeout: int=60, k: Union[int, list]=1, **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/humaneval')
        self.k = k
        super().__init__(name=type(self).__name__, path=path, mode=mode, timeout=timeout, **kwargs)

    def _load_data(self):
        data_path = os.path.join(self.path, 'HumanEval.jsonl')
        if not os.path.exists(data_path):
            download_raw_humaneval_data(self.path)
        if self.mode == 'train' or self.mode == 'all':
            self._train_data = None
        if self.mode == 'dev' or self.mode == 'all':
            self._dev_data = None
        if self.mode == 'test' or self.mode == 'all':
            self._test_data = load_humaneval_data(data_path)

    def _get_label(self, example: Any):
        return {'task_id': example['task_id'], 'canonical_solution': example['canonical_solution'], 'test': example['test'], 'entry_point': example['entry_point']}

    def _get_id(self, example: Any):
        return example['task_id']

    def handle_special_cases(self, task_id: str, solution: str, test: str) -> bool:
        """
        Handle special cases for HumanEval.
        """
        if task_id == 'HumanEval/50':
            solution = '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n' + solution
            return (solution, test)
        return super().handle_special_cases(task_id=task_id, solution=solution, test=test)

    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Evaluate the solution code.

        Args:
            prediction (str | List[str]): The solution code(s).
            label (dict | List[dict]): The unit test code(s).

        Returns:
            dict: The evaluation metrics (pass@k).
        """
        prediction, label = self._check_evaluation_inputs(prediction, label)
        results = []
        for solution in prediction:
            solution_states = []
            for label_data in label:
                task_id = label_data['task_id']
                prompt = self.get_example_by_id(task_id)['prompt']
                unit_test = label_data['test']
                entry_point = label_data['entry_point']
                state, message = self.check_solution(task_id=task_id, solution=prompt + solution, test=unit_test, entry_point=entry_point)
                if state != self.SUCCESS:
                    break
                solution_states.append(state)
            results.append(len(solution_states) == len(label) and all((state == self.SUCCESS for state in solution_states)))
        k_list = [self.k] if isinstance(self.k, int) else self.k
        pass_at_k = self.compute_pass_at_k(results, k_list)
        return pass_at_k

def load_humaneval_data(data_path: str):
    data = load_json(data_path, type='jsonl')
    for example in data:
        if example['task_id'] == 'HumanEval/115':
            example['prompt'] = 'import math\n' + example['prompt'].replace('import math', '')
    return data

def download_raw_hotpotqa_data(name: str, save_folder: str):
    assert name in VALIDE_RAW_HOTPOTQA_FILES, f"'{name}' is an invalid hotpotqa file name. Available file names: {VALIDE_RAW_HOTPOTQA_FILES}"
    url = f'http://curtis.ml.cmu.edu/datasets/hotpot/{name}'
    typ = 'train' if 'train' in name else 'dev'
    logger.info(f'Downloading HotPotQA {typ} data from: {url}')
    download_file(url=url, save_file=os.path.join(save_folder, name))

class HotPotQA(Benchmark):
    """Benchmark class for evaluating multi-hop question answering on HotPotQA dataset.
    
    Each HotPotQA example has the following structure:
    {
        "_id": str, 
        "question": str, 
        "answer": str, 
        "context": [["context_title", ["context_sentence", "another_sentence"]]],
        "supporting_facts": [["supporting_title", supporting_sentence_index]],
        "type": str,
        "level": str
    }
    
    The benchmark evaluates answers using exact match, F1 score, and accuracy metrics.
    """

    def __init__(self, path: str=None, mode: str='all', **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/hotpotqa')
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_hotpotqa_data(name=file_name, save_folder=self.path)
        logger.info(f'loading HotPotQA data from {file_path} ...')
        return load_json(path=file_path, type='json')

    def _load_data(self):
        if self.mode == 'train' or self.mode == 'all':
            self._train_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP['train'])
        if self.mode == 'dev' or self.mode == 'all':
            self._dev_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP['dev'])
        if self.mode == 'test' or self.mode == 'all':
            self._test_data = self._load_data_from_file(file_name=HOTPOTQA_FILES_MAP['test'])

    def _get_label(self, example: Any) -> Any:
        return example['answer']

    def _get_id(self, example: Any) -> Any:
        return example['_id']

    def evaluate(self, prediction: Any, label: Any) -> dict:
        em = exact_match_score(prediction=prediction, ground_truth=label)
        f1 = f1_score(prediction=prediction, ground_truth=label)
        acc = acc_score(prediction=prediction, ground_truths=[label])
        return {'f1': f1, 'em': em, 'acc': acc}

def download_raw_nq_data(name: str, save_folder: str):
    assert name in VALID_RAW_NQ_FILES, f"'{name}' is an invalid nq file name. Available file names: {VALID_RAW_NQ_FILES}"
    file_type_map = {file_name: typ for typ, file_name in NQ_FILES_MAP.items()}
    typ = file_type_map[name]
    url = f'https://dl.fbaipublicfiles.com/dpr/data/retriever/{name}'
    logger.info(f'Downloading NQ {typ} data from: {url}')
    download_file(url=url, save_file=os.path.join(save_folder, name))

class NQ(Benchmark):
    """Benchmark class for evaluating question answering on Natural Questions dataset.
    
    Natural Questions (NQ) is a dataset for open-domain question answering,
    containing real questions from Google Search and answers from Wikipedia.
    This class handles loading the dataset, evaluating answers, and computing
    metrics like exact match and F1 score.
    
    Each NQ example has the following structure:
    {
        "id": str, 
        "question": str, 
        "answers": List[str]
    }
    
    The benchmark evaluates answers using exact match, F1 score, and accuracy metrics.
    """

    def __init__(self, path: str=None, mode: str='all', **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/nq')
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_nq_data(name=file_name, save_folder=self.path)
        logger.info(f'loading NQ data from {file_path} ...')
        return load_tsv_data(file_path=file_path)

    def _load_data(self):
        if self.mode == 'train' or self.mode == 'all':
            self._train_data = self._load_data_from_file(file_name=NQ_FILES_MAP['train'])
        if self.mode == 'dev' or self.mode == 'all':
            self._dev_data = self._load_data_from_file(file_name=NQ_FILES_MAP['dev'])
        if self.mode == 'test' or self.mode == 'all':
            self._test_data = self._load_data_from_file(file_name=NQ_FILES_MAP['test'])

    def _get_label(self, example: Any) -> Any:
        return example['answers']

    def _get_id(self, example: Any) -> Any:
        return example['id']

    def evaluate(self, prediction: Any, label: Any) -> dict:
        em = ems(prediction=prediction, ground_truths=label)
        f1 = max((f1_score(prediction=prediction, ground_truth=one_answer) for one_answer in label))
        acc = acc_score(prediction=prediction, ground_truths=label)
        return {'f1': f1, 'em': em, 'acc': acc}

def load_tsv_data(file_path: str) -> List[dict]:
    base_name = os.path.basename(file_path)
    file_type_map = {file_name: typ for typ, file_name in NQ_FILES_MAP.items()}
    assert base_name in file_type_map, f"'{base_name}' is an invalid nq file name. Available file names: {VALID_RAW_NQ_FILES}"
    typ = file_type_map[base_name]
    data = []
    with open(file_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            question, answers = line.strip().split('\t')
            answers = eval(answers)
            data.append({'id': f'{typ}-{i + 1}', 'question': question, 'answers': answers})
    return data

def download_raw_gsm8k_data(name: str, save_folder: str):
    assert name in VALID_RAW_GSM8K_FILES, f"'{name}' is an invalid GSM8K file name. Available file names: {VALID_RAW_GSM8K_FILES}"
    url = f'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{name}'
    typ = 'train' if 'train' in name else 'test'
    logger.info(f'Downloading GSM8K {typ} data from: {url}')
    download_file(url=url, save_file=os.path.join(save_folder, name))

class GSM8K(Benchmark):
    """Benchmark class for evaluating math reasoning on GSM8K dataset.
    
    GSM8K (Grade School Math 8K) is a dataset of math word problems that
    test a model's ability to solve grade school level math problems requiring
    multi-step reasoning. This class handles loading the dataset, evaluating
    solutions, and computing metrics based on answer accuracy.
    
    Each GSM8K example has the following structure:
    {
        "id": "test-1", 
        "question": "the question", 
        "answer": "the answer"
    }
    
    The benchmark evaluates answers by extracting the final numerical value
    and comparing it to the ground truth answer.
    """

    def __init__(self, path: str=None, mode: str='all', **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/gsm8k')
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_raw_gsm8k_data(name=file_name, save_folder=self.path)
        logger.info(f'loading GSM8K data from {file_path} ...')
        return load_gsm8k_data(file_path=file_path)

    def _load_data(self):
        if self.mode == 'train' or self.mode == 'all':
            self._train_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP['train'])
        if self.mode == 'dev' or self.mode == 'all':
            self._dev_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP['dev'])
        if self.mode == 'test' or self.mode == 'all':
            self._test_data = self._load_data_from_file(file_name=GSM8K_FILES_MAP['test'])

    def _get_label(self, example: Any) -> Any:
        return example['answer']

    def _get_id(self, example: Any) -> Any:
        return example['id']

    def extract_last_number(self, text: str) -> float:
        """
        Extract the last number from a text.
        """
        matches = regex.findall('[-+]?\\d+(?:,\\d{3})*(?:\\.\\d+)?|\\d+\\.\\d+', str(text))
        if matches:
            last_number = matches[-1].replace(',', '').strip()
            try:
                last_number = float(last_number)
                return last_number
            except ValueError:
                return None
        return None

    def evaluate(self, prediction: Any, label: Any) -> dict:
        ground_truth_answer = self.extract_last_number(label)
        predicted_answer = self.extract_last_number(prediction)
        if predicted_answer is None:
            return {'solve_rate': 0.0}
        solve_rate = 1.0 if abs(predicted_answer - ground_truth_answer) < 1e-06 else 0.0
        return {'solve_rate': solve_rate}

def load_gsm8k_data(file_path: str) -> List[dict]:
    base_name = os.path.basename(file_path)
    file_type_map = {file_name: typ for typ, file_name in GSM8K_FILES_MAP.items()}
    assert base_name in file_type_map, f"'{base_name}' is an invalid gsm8k file name. Available file names: {VALID_RAW_GSM8K_FILES}"
    typ = file_type_map[base_name]
    data = load_json(path=file_path, type='jsonl')
    new_data = []
    for i, example in enumerate(data):
        item = {'id': f'{typ}-{i + 1}'}
        item.update(example)
        new_data.append(item)
    return new_data


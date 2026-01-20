# Cluster 11

def make_parent_folder(path: str):
    """Checks if the parent folder of a given path exists, and creates it if not.

    Args:
        path (str): The file path for which to create the parent folder.
    """
    dir_folder = os.path.dirname(path)
    if dir_folder and (not os.path.exists(dir_folder)):
        logger.info(f'creating folder {dir_folder} ...')
        os.makedirs(dir_folder, exist_ok=True)

def download_aflow_benchmark_data(dataset: str, save_folder: str):
    candidate_datasets = list(AFLOW_DATASET_FILES_MAP.keys()) + ['all']
    lower_candidate_datasets = [dataset.lower() for dataset in candidate_datasets]
    if dataset.lower() not in lower_candidate_datasets:
        raise ValueError(f'Invalid value for dataset: {dataset}. Available choices: {candidate_datasets}')
    url = 'https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e'
    logger.info(f'Downloading AFlow benchmark data from {url} ...')
    aflow_data_save_file = os.path.join(save_folder, 'aflow_data.tar.gz')
    make_parent_folder(aflow_data_save_file)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(aflow_data_save_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    logger.info(f'Extracting data for {dataset} dataset(s) from {aflow_data_save_file} ...')
    extract_tar_gz(aflow_data_save_file, extract_path=save_folder)
    if dataset != 'all':
        dataset_files = [file for file in list(AFLOW_DATASET_FILES_MAP[dataset].values()) if file is not None]
        for file in os.listdir(save_folder):
            if file not in dataset_files:
                os.remove(os.path.join(save_folder, file))
    if os.path.exists(aflow_data_save_file):
        logger.info(f'Remove {aflow_data_save_file}')
        os.remove(aflow_data_save_file)

def extract_tar_gz(filename: str, extract_path: str) -> None:
    """Extract a tar.gz file to the specified path."""
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(path=extract_path)

class AFlowMBPP(MBPP):
    """
    AFlow-specific implementation of MBPP benchmark.
    """

    def __init__(self, path: str=None, mode: str='all', timeout: int=60, k: Union[int, list]=1, **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/aflow/mbpp')
        super().__init__(path=path, mode=mode, timeout=timeout, k=k, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset='mbpp', save_folder=self.path)
        return load_json(path=file_path, type='jsonl')

    def _load_data(self):
        if self.mode == 'train' or self.mode == 'all':
            logger.info(f'Loading train data from {AFLOW_DATASET_FILES_MAP['mbpp']['train']}')
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['mbpp']['train'])
        if self.mode == 'dev' or self.mode == 'all':
            logger.info(f'Loading dev data from {AFLOW_DATASET_FILES_MAP['mbpp']['dev']}')
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['mbpp']['dev'])
        if self.mode == 'test' or self.mode == 'all':
            logger.info(f'Loading test data from {AFLOW_DATASET_FILES_MAP['mbpp']['test']}')
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['mbpp']['test'])
        self._test_cases = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['mbpp']['test_cases'])

    def _get_label(self, example: Any):
        return {'task_id': example['task_id'], 'canonical_solution': example['code'], 'test': example['test'], 'entry_point': example['entry_point']}

    def extract_test_cases_with_entry_point(self, entry_point: str):
        hardcoded_cases = {'remove_odd': '', 'replace_spaces': '', 'snake_to_camel': '', 'Split': '', 'swap_List': '', 'square_Sum': '', 'sort_sublists': '', 'unique_sublists': ''}
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]
        for case in self._test_cases:
            if case['entry_point'] == entry_point:
                return case['test']
        return None

    async def async_evaluate(self, graph: Callable, example: Any) -> float:
        prompt, entry_point = (example['prompt'], example['entry_point'])
        solution = await graph(prompt, entry_point)
        label = self._get_label(example)
        metrics = await super().async_evaluate(prediction=solution, label=label)
        return metrics['pass@1']

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
                state, message = self.check_solution(task_id=task_id, solution=prompt + '\n' + solution, test=unit_test, entry_point=entry_point, use_entrypoint_as_input=False)
                if state != self.SUCCESS:
                    break
                solution_states.append(state)
            results.append(len(solution_states) == len(label) and all((state == self.SUCCESS for state in solution_states)))
        k_list = [self.k] if isinstance(self.k, int) else self.k
        pass_at_k = self.compute_pass_at_k(results, k_list)
        return pass_at_k

class AFlowHumanEval(HumanEval):
    """
    AFlow-specific implementation of HumanEval benchmark.
    """

    def __init__(self, path: str=None, mode: str='all', timeout: int=60, k: Union[int, list]=1, **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/aflow/humaneval')
        super().__init__(path=path, mode=mode, timeout=timeout, k=k, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset='humaneval', save_folder=self.path)
        return load_json(path=file_path, type='jsonl')

    def _load_data(self):
        if self.mode == 'train' or self.mode == 'all':
            logger.info(f'Loading train data from {AFLOW_DATASET_FILES_MAP['humaneval']['train']}')
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['humaneval']['train'])
        if self.mode == 'dev' or self.mode == 'all':
            logger.info(f'Loading dev data from {AFLOW_DATASET_FILES_MAP['humaneval']['dev']}')
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['humaneval']['dev'])
        if self.mode == 'test' or self.mode == 'all':
            logger.info(f'Loading test data from {AFLOW_DATASET_FILES_MAP['humaneval']['test']}')
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['humaneval']['test'])
        self._test_cases = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['humaneval']['test_cases'])

    def extract_test_cases_with_entry_point(self, entry_point: str):
        """
        Extract test cases with the given entry point.
        """
        hardcoded_cases = {'find_zero': '', 'decode_cyclic': '', 'decode_shift': '', 'by_length': '', 'add': '', 'triangle_area': '', 'correct_bracketing': '', 'solve': '', 'sum_squares': '', 'starts_one_ends': ''}
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]
        for case in self._test_cases:
            if case['entry_point'] == entry_point:
                return case['test']
        return None

    async def async_evaluate(self, graph: Callable, example: Any) -> float:
        prompt, entry_point = (example['prompt'], example['entry_point'])
        solution = await graph(prompt, entry_point)
        label = self._get_label(example)
        metrics = await super().async_evaluate(prediction=solution, label=label)
        return metrics['pass@1']

def download_raw_math_data(save_folder: str):
    """
    Download the MATH data from the modelscope website.
    """
    url = 'https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip'
    logger.info(f'Downloading MATH data from {url} ...')
    save_file_path = os.path.join(save_folder, 'MATH.zip')
    make_parent_folder(save_file_path)
    if not os.path.exists(save_file_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    with zipfile.ZipFile(save_file_path, 'r') as zip_ref:
        zip_ref.extractall(save_folder)
    if os.path.exists(save_file_path):
        os.remove(save_file_path)

class MATH(Benchmark):
    """Benchmark class for evaluating mathematical reasoning on the MATH dataset.
    
    MATH is a dataset of challenging competition mathematics problems,
    spanning various difficulty levels and subject areas. This class handles
    loading the dataset, extracting answers, evaluating solutions through
    symbolic and numerical comparisons, and computing accuracy metrics.
    
    The dataset includes problems across 7 subject areas (Algebra, Geometry, etc.)
    and 5 difficulty levels. Each problem contains LaTeX-formatted
    questions and solutions.
    
    Each MATH example has the following structure:
    {
        "id": "test-1", 
        "problem": "the problem", 
        "solution": "the solution",
        "level": "Level 1", # "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level ?"
        "type": "Algebra", # 'Geometry', 'Algebra', 'Intermediate Algebra', 'Counting & Probability', 'Precalculus', 'Number Theory', 'Prealgebra'
    }
    
    The benchmark evaluates answers using symbolic math equality checking
    and numerical approximation to handle equivalent mathematical expressions.
    """

    def __init__(self, path: str=None, mode: str='all', **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/math')
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_folders(self, data_folder: str) -> List[dict]:
        if data_folder is None:
            return None
        data = []
        typ = 'train' if 'train' in data_folder else 'test'
        sub_data_folders = os.listdir(data_folder)
        i = 0
        logger.info(f'loading MATH data from {data_folder} ...')
        for sub_data_folder in sub_data_folders:
            if os.path.isdir(os.path.join(data_folder, sub_data_folder)):
                files = os.listdir(os.path.join(data_folder, sub_data_folder))
                for file in files:
                    if file.endswith('.json'):
                        example = {'id': f'{typ}-{i + 1}'}
                        example.update(load_json(os.path.join(data_folder, sub_data_folder, file), type='json'))
                        data.append(example)
                        i += 1
        return data

    def _load_data(self):
        if not os.path.exists(os.path.join(self.path, 'MATH')):
            download_raw_math_data(save_folder=self.path)
        data_folder = os.path.join(self.path, 'MATH')
        if self.mode == 'train' or self.mode == 'all':
            self._train_data = self._load_data_from_folders(data_folder=os.path.join(data_folder, 'train'))
        if self.mode == 'dev' or self.mode == 'all':
            self._dev_data = None
        if self.mode == 'test' or self.mode == 'all':
            self._test_data = self._load_data_from_folders(data_folder=os.path.join(data_folder, 'test'))

    def _get_label(self, example: Any) -> Any:
        return example['solution']

    def _get_id(self, example: Any) -> Any:
        return example['id']

    def extract_answer(self, text: str) -> str:
        pattern = '\\\\boxed{((?:[^{}]|{[^{}]*})*)}'
        boxed_matches = regex.findall(pattern, text, regex.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()
        sentence_end_pattern = '(?<!\\d)[.!?]\\s+'
        sentences = regex.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ''

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True
        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=0.001)
        except Exception:
            pass
        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass
        return False

    def is_digit(self, num: Any) -> bool:
        return self.parse_digits(num) is not None

    def parse_digits(self, num: Any) -> float:
        num = regex.sub(',', '', str(num))
        try:
            return float(num)
        except Exception:
            if num.endswith('%'):
                num = num[:-1]
                if num.endswith('\\'):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except Exception:
                    pass
        return None

    def symbolic_equal(self, a: Any, b: Any) -> bool:

        def _parse(s: Any) -> Any:
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except Exception:
                    pass
            return s
        a = _parse(a)
        b = _parse(b)
        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass
        try:
            if isclose(N(a), N(b), abs_tol=0.001):
                return True
        except Exception:
            pass
        return False

    def evaluate(self, prediction: Any, label: Any) -> dict:
        ground_truth_answer = self.extract_answer(label)
        predicted_answer = self.extract_answer(prediction)
        solve_rate = 1.0 if self.math_equal(predicted_answer, ground_truth_answer) else 0.0
        return {'solve_rate': solve_rate}

class AFlowMATH(MATH):

    def __init__(self, path: str=None, mode: str='all', **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/aflow/math')
        super().__init__(path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset='math', save_folder=self.path)
        return load_json(path=file_path, type='jsonl')

    def _load_data(self):
        if self.mode == 'train' or self.mode == 'all':
            logger.info(f'Loading train data from {AFLOW_DATASET_FILES_MAP['math']['train']}')
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['math']['train'])
        if self.mode == 'dev' or self.mode == 'all':
            logger.info(f'Loading dev data from {AFLOW_DATASET_FILES_MAP['math']['dev']}')
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['math']['dev'])
        if self.mode == 'test' or self.mode == 'all':
            logger.info(f'Loading test data from {AFLOW_DATASET_FILES_MAP['math']['test']}')
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['math']['test'])

    async def async_evaluate(self, graph: Callable, example: Any) -> float:
        problem = example['problem']
        label = self._get_label(example)
        output = await graph(problem)
        metrics = await super().async_evaluate(prediction=output, label=label)
        return metrics['solve_rate']

class AFlowHotPotQA(HotPotQA):
    """
    AFlow-specific implementation of HotPotQA benchmark.
    """

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset='hotpotqa', save_folder=self.path)
        logger.info(f'loading data from {file_path} ...')
        return load_json(path=file_path, type='jsonl')

    def _load_data(self):
        if self.mode == 'train' or self.mode == 'all':
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['hotpotqa']['train'])
        if self.mode == 'dev' or self.mode == 'all':
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['hotpotqa']['dev'])
        if self.mode == 'test' or self.mode == 'all':
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['hotpotqa']['test'])

    async def async_evaluate(self, graph: Callable, example: Any) -> float:
        prompt = example['question']
        paragraphs = [item[1] for item in example['context'] if isinstance(item[1], list)]
        context_str = '\n'.join((' '.join(paragraph) for paragraph in paragraphs))
        inputs = f'Context: {context_str}\n\nQuestion: {prompt}\n\nAnswer:'
        solution = await graph(inputs)
        label = self._get_label(example)
        metrics = await super().async_evaluate(prediction=solution, label=label)
        return metrics['f1']

class AFlowGSM8K(GSM8K):
    """AFlow-specific implementation of GSM8K benchmark.
    
    This class extends the GSM8K benchmark with features specific to the
    AFlow framework, including loading from AFlow-formatted data files and
    supporting asynchronous evaluation for workflows.
    
    Attributes:
        path: Path to the directory containing AFlow-formatted GSM8K files.
        mode: Data loading mode ("train", "dev", "test", or "all").
        _train_data: Training dataset loaded from AFlow format.
        _dev_data: Development dataset loaded from AFlow format.
        _test_data: Test dataset loaded from AFlow format.
    """

    def __init__(self, path: str=None, mode: str='all', **kwargs):
        path = os.path.expanduser(path or '~/.evoagentx/data/aflow/gsm8k')
        super().__init__(path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        if file_name is None:
            return None
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            download_aflow_benchmark_data(dataset='gsm8k', save_folder=self.path)
        return load_json(path=file_path, type='jsonl')

    def _load_data(self):
        if self.mode == 'train' or self.mode == 'all':
            logger.info(f'Loading train data from {AFLOW_DATASET_FILES_MAP['gsm8k']['train']}')
            self._train_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['gsm8k']['train'])
        if self.mode == 'dev' or self.mode == 'all':
            logger.info(f'Loading dev data from {AFLOW_DATASET_FILES_MAP['gsm8k']['dev']}')
            self._dev_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['gsm8k']['dev'])
        if self.mode == 'test' or self.mode == 'all':
            logger.info(f'Loading test data from {AFLOW_DATASET_FILES_MAP['gsm8k']['test']}')
            self._test_data = self._load_data_from_file(file_name=AFLOW_DATASET_FILES_MAP['gsm8k']['test'])

    async def async_evaluate(self, graph: Callable, example: Any) -> float:
        input_text = example['question']
        label = self._get_label(example)
        output = await graph(input_text)
        metrics = await super().async_evaluate(prediction=output, label=label)
        return metrics['solve_rate']

def save_json(data, path: str, type: str='json', use_indent: bool=True) -> str:
    """
    save data to a json file

    Args: 
        data: The json data to be saved. It can be a JSON str or a Serializable object when type=="json" or a list of JSON str or Serializable object when type=="jsonl".
        path(str): The path of the saved json file. 
        type(str): The type of the json file, chosen from ["json" or "jsonl"].
        use_indent: Whether to use indent when saving the json file. 
    
    Returns:
        path: the path where the json data is saved. 
    """
    assert type in ['json', 'jsonl']
    make_parent_folder(path)
    if type == 'json':
        with open(path, 'w', encoding='utf-8') as fout:
            if use_indent:
                fout.write(data if isinstance(data, str) else json.dumps(data, indent=4))
            else:
                fout.write(data if isinstance(data, str) else json.dumps(data))
    elif type == 'jsonl':
        with open(path, 'w', encoding='utf-8') as fout:
            for item in data:
                fout.write('{}\n'.format(item if isinstance(item, str) else json.dumps(item)))
    return path

class BaseModule(BaseModel, metaclass=MetaModule):
    """
    Base module class that serves as the foundation for all modules in the EvoAgentX framework.
    
    This class provides serialization/deserialization capabilities, supports creating instances from
    dictionaries, JSON, or files, and exporting instances to these formats.
    
    Attributes:
        class_name: The class name, defaults to None but is automatically set during subclass initialization
        model_config: Pydantic model configuration that controls type matching and behavior
    """
    class_name: str = None
    model_config = {'arbitrary_types_allowed': True, 'extra': 'allow', 'protected_namespaces': (), 'validate_assignment': False}

    def __init_subclass__(cls, **kwargs):
        """
        Subclass initialization method that automatically sets the class_name attribute.
        
        Args:
            cls (Type): The subclass being initialized
            **kwargs (Any): Additional keyword arguments
        """
        super().__init_subclass__(**kwargs)
        cls.class_name = cls.__name__

    def __init__(self, **kwargs):
        """
        Initializes a BaseModule instance.
        
        Args:
            **kwargs (Any): Keyword arguments used to initialize the instance
        
        Raises:
            ValidationError: When parameter validation fails
            Exception: When other errors occur during initialization
        """
        try:
            for field_name, _ in type(self).model_fields.items():
                field_value = kwargs.get(field_name, None)
                if field_value:
                    kwargs[field_name] = self._process_data(field_value)
            super().__init__(**kwargs)
            self.init_module()
        except (ValidationError, Exception) as e:
            exception_handler = callback_manager.get_callback('exception_buffer')
            if exception_handler is None:
                error_message = get_base_module_init_error_message(cls=self.__class__, data=kwargs, errors=e)
                logger.error(error_message)
                raise
            else:
                exception_handler.add(e)

    def init_module(self):
        """
        Module initialization method that subclasses can override to provide additional initialization logic.
        """
        pass

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        
        Returns:
            str: String representation of the object
        """
        return self.to_str()

    @property
    def kwargs(self) -> dict:
        """
        Returns the extra fields of the model.
        
        Returns:
            dict: Dictionary containing all extra keyword arguments
        """
        return self.model_extra

    @classmethod
    def _create_instance(cls, data: Dict[str, Any]) -> 'BaseModule':
        """
        Internal method for creating an instance from a dictionary.
        
        Args:
            data: Dictionary containing instance data
        
        Returns:
            BaseModule: The created instance
        """
        processed_data = {k: cls._process_data(v) for k, v in data.items()}
        return cls.model_validate(processed_data)

    @classmethod
    def _process_data(cls, data: Any) -> Any:
        """
        Recursive method for processing data, with special handling for dictionaries containing class_name.
        
        Args:
            data: Data to be processed
        
        Returns:
            Processed data
        """
        if isinstance(data, dict):
            if 'class_name' in data:
                sub_class = MODULE_REGISTRY.get_module(data.get('class_name'))
                return sub_class._create_instance(data)
            else:
                return {k: cls._process_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [cls._process_data(x) for x in data]
        else:
            return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> 'BaseModule':
        """
        Instantiate the BaseModule from a dictionary.
        
        Args:
            data: Dictionary containing instance data
            **kwargs (Any): Additional keyword arguments, can include log to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            Exception: When errors occur during initialization
        """
        use_logger = kwargs.get('log', True)
        with exception_buffer() as buffer:
            try:
                class_name = data.get('class_name', None)
                if class_name:
                    cls = MODULE_REGISTRY.get_module(class_name)
                module = cls._create_instance(data)
                if len(buffer.exceptions) > 0:
                    error_message = get_base_module_init_error_message(cls, data, buffer.exceptions)
                    if use_logger:
                        logger.error(error_message)
                    raise Exception(get_error_message(buffer.exceptions))
            finally:
                pass
        return module

    @classmethod
    def from_json(cls, content: str, **kwargs) -> 'BaseModule':
        """
        Construct the BaseModule from a JSON string.
        
        This method uses yaml.safe_load to parse the JSON string into a Python object,
        which supports more flexible parsing than standard json.loads (including handling
        single quotes, trailing commas, etc). The parsed data is then passed to from_dict
        to create the instance.
        
        Args:
            content: JSON string
            **kwargs (Any): Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the input is not a valid JSON string
        """
        use_logger = kwargs.get('log', True)
        try:
            data = yaml.safe_load(content)
        except Exception:
            error_message = f'Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_json is not a valid JSON string.'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        if not isinstance(data, (list, dict)):
            error_message = f'Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_json is not a valid JSON string.'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        return cls.from_dict(data, log=use_logger)

    @classmethod
    def from_str(cls, content: str, **kwargs) -> 'BaseModule':
        """
        Construct the BaseModule from a string that may contain JSON.
        
        This method is more forgiving than `from_json` as it can extract valid JSON
        objects embedded within larger text. It uses `parse_json_from_text` to extract 
        all potential JSON strings from the input text, then tries to create an instance 
        from each extracted JSON string until successful.
        
        Args:
            content: Text that may contain JSON strings
            **kwargs (Any): Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the input does not contain valid JSON strings or the JSON is incompatible with the class
        """
        use_logger = kwargs.get('log', True)
        extracted_json_list = parse_json_from_text(content)
        if len(extracted_json_list) == 0:
            error_message = f'The input to {cls.__name__}.from_str does not contain any valid JSON str.'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        module = None
        for json_str in extracted_json_list:
            try:
                module = cls.from_json(json_str, log=False)
            except Exception:
                continue
            break
        if module is None:
            error_message = f'Can not instantiate {cls.__name__}. The input to {cls.__name__}.from_str either does not contain a valide JSON str, or the JSON str is incomplete or incompatable (incorrect variables or types) with {cls.__name__}.'
            error_message += f'\nInput:\n{content}'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        return module

    @classmethod
    def load_module(cls, path: str, **kwargs) -> dict:
        """
        Load the values for a module from a file.
        
        By default, it opens the specified file and uses `yaml.safe_load` to parse its contents 
        into a Python object (typically a dictionary).
        
        Args:
            path: The path of the file
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            dict: The JSON object instantiated from the file
        """
        with open(path, mode='r', encoding='utf-8') as file:
            content = yaml.safe_load(file.read())
        return content

    @classmethod
    def from_file(cls, path: str, load_function: Callable=None, **kwargs) -> 'BaseModule':
        """
        Construct the BaseModule from a file.
        
        This method reads and parses a file into a data structure, then creates
        a module instance from that data. It first verifies that the file exists,
        then uses either the provided `load_function` or the default `load_module`
        method to read and parse the file content, and finally calls `from_dict`
        to create the instance.
        
        Args:
            path: The path of the file
            load_function: The function used to load the data, takes a file path as input and returns a JSON object
            **kwargs (Any): Additional keyword arguments, can include `log` to control logging output
        
        Returns:
            BaseModule: The created module instance
        
        Raises:
            ValueError: When the file does not exist
        """
        use_logger = kwargs.get('log', True)
        if not os.path.exists(path):
            error_message = f'File "{path}" does not exist!'
            if use_logger:
                logger.error(error_message)
            raise ValueError(error_message)
        function = load_function or cls.load_module
        content = function(path, **kwargs)
        module = cls.from_dict(content, log=use_logger)
        return module

    def to_dict(self, exclude_none: bool=True, ignore: List[str]=[], **kwargs) -> dict:
        """
        Convert the BaseModule to a dictionary.
        
        Args:
            exclude_none: Whether to exclude fields with None values
            ignore: List of field names to ignore
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            dict: Dictionary containing the object data
        """
        data = {}
        for field_name, _ in type(self).model_fields.items():
            if field_name in ignore:
                continue
            field_value = getattr(self, field_name, None)
            if exclude_none and field_value is None:
                continue
            if isinstance(field_value, BaseModule):
                data[field_name] = field_value.to_dict(exclude_none=exclude_none, ignore=ignore)
            elif isinstance(field_value, list):
                data[field_name] = [item.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(item, BaseModule) else item for item in field_value]
            elif isinstance(field_value, dict):
                data[field_name] = {key: value.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(value, BaseModule) else value for key, value in field_value.items()}
            else:
                data[field_name] = field_value
        return data

    def to_json(self, use_indent: bool=False, ignore: List[str]=[], **kwargs) -> str:
        """
        Convert the BaseModule to a JSON string.
        
        Args:
            use_indent: Whether to use indentation
            ignore: List of field names to ignore
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            str: The JSON string
        """
        if use_indent:
            kwargs['indent'] = kwargs.get('indent', 4)
        else:
            kwargs.pop('indent', None)
        if kwargs.get('default', None) is None:
            kwargs['default'] = custom_serializer
        data = self.to_dict(exclude_none=True)
        for ignore_field in ignore:
            data.pop(ignore_field, None)
        return json.dumps(data, **kwargs)

    def to_str(self, **kwargs) -> str:
        """
        Convert the BaseModule to a string. Use .to_json to output JSON string by default.
        
        Args:
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            str: The string
        """
        return self.to_json(use_indent=False)

    def save_module(self, path: str, ignore: List[str]=[], **kwargs) -> str:
        """
        Save the BaseModule to a file.
        
        This method will set non-serializable objects to None by default.
        If you want to save non-serializable objects, override this method.
        Remember to also override the `load_module` function to ensure the loaded
        object can be correctly parsed by `cls.from_dict`.
        
        Args:
            path: The path to save the file
            ignore: List of field names to ignore
            **kwargs (Any): Additional keyword arguments
        
        Returns:
            str: The path where the file is saved, same as the input path
        """
        logger.info('Saving {} to {}', self.__class__.__name__, path)
        return save_json(self.to_json(use_indent=True, default=lambda x: None, ignore=ignore), path=path)

    def deepcopy(self):
        """Deep copy the module.

        This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other
        attributes, we just do the shallow copy.
        """
        try:
            return copy.deepcopy(self)
        except Exception:
            pass
        new_instance = self.__class__.__new__(self.__class__)
        for attr, value in self.__dict__.items():
            if isinstance(value, BaseModule):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    logging.warning(f"Failed to deep copy attribute '{attr}' of {self.__class__.__name__}, falling back to shallow copy or reference copy.")
                    try:
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        setattr(new_instance, attr, value)
        return new_instance

class TestMBPP(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = load_json(path='tests/data/benchmark/mbpp_samples.json', type='json')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        test_file = os.path.join(self.temp_dir, 'sanitized-mbpp.json')
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        save_json(self.sample_data, test_file, type='json')

    @patch('evoagentx.benchmark.mbpp.download_raw_mbpp_data')
    def test_load_data(self, mock_download):
        self.create_test_files()
        benchmark = MBPP(path=self.temp_dir)
        self.assertEqual(len(benchmark.get_train_data()), 0)
        self.assertEqual(len(benchmark.get_dev_data()), 0)
        self.assertEqual(len(benchmark.get_test_data()), 10)
        self.assertEqual(mock_download.call_count, 0)

    def test_get_label(self):
        self.create_test_files()
        benchmark = MBPP(path=self.temp_dir, mode='test')
        example = benchmark.get_test_data()[0]
        label = benchmark.get_label(example)
        self.assertTrue(isinstance(label, dict))
        self.assertEqual(label['task_id'], self.sample_data[0]['task_id'])
        self.assertEqual(label['canonical_solution'], self.sample_data[0]['code'])
        for i, example in enumerate(benchmark.get_test_data()):
            label = benchmark.get_label(example)
            self.assertTrue(isinstance(label, dict))
            self.assertEqual(label['task_id'], self.sample_data[i]['task_id'])
            self.assertEqual(label['canonical_solution'], self.sample_data[i]['code'])
            entry_point = label['entry_point']
            test = label['test']
            self.assertTrue(all((entry_point in assert_str for assert_str in self.sample_data[i]['test_list'])))
            self.assertTrue(all((assert_str in test for assert_str in self.sample_data[i]['test_list'])))

    def test_evaluate(self):
        self.create_test_files()
        benchmark = MBPP(path=self.temp_dir, mode='test')
        test_data = benchmark.get_test_data()
        for example in test_data:
            prediction = example['canonical_solution']
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(prediction, label)
            self.assertEqual(len(metrics), 1)
            self.assertTrue('pass@1' in metrics)
            self.assertTrue(metrics['pass@1'] == 1.0)

class TestHumanEval(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = load_json(path='tests/data/benchmark/humaneval_samples.jsonl', type='jsonl')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        test_file = os.path.join(self.temp_dir, 'HumanEval.jsonl')
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        save_json(self.sample_data, test_file, type='jsonl')

    @patch('evoagentx.benchmark.humaneval.download_raw_humaneval_data')
    def test_load_data(self, mock_download):
        self.create_test_files()
        benchmark = HumanEval(path=self.temp_dir)
        self.assertEqual(len(benchmark.get_train_data()), 0)
        self.assertEqual(len(benchmark.get_dev_data()), 0)
        self.assertEqual(len(benchmark.get_test_data()), 10)
        self.assertEqual(mock_download.call_count, 0)

    def test_get_label(self):
        self.create_test_files()
        benchmark = HumanEval(path=self.temp_dir, mode='test')
        example = benchmark.get_test_data()[0]
        label = benchmark.get_label(example)
        self.assertTrue(isinstance(label, dict))
        self.assertEqual(label['task_id'], self.sample_data[0]['task_id'])
        self.assertEqual(label['canonical_solution'], self.sample_data[0]['canonical_solution'])
        self.assertEqual(label['test'], self.sample_data[0]['test'])
        self.assertEqual(label['entry_point'], self.sample_data[0]['entry_point'])

    def test_evaluate(self):
        self.create_test_files()
        benchmark = HumanEval(path=self.temp_dir, mode='test')
        test_data = benchmark.get_test_data()
        for example in test_data:
            prediction = example['prompt'] + example['canonical_solution']
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(prediction, label)
            self.assertEqual(len(metrics), 1)
            self.assertTrue('pass@1' in metrics)
            self.assertEqual(metrics['pass@1'], 1.0)

class TestMath(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data1 = {'problem': 'We roll a fair 6-sided die 5 times.  What is the probability that we get a 6 in at most 2 of the rolls?', 'level': 'Level 5', 'type': 'Counting & Probability', 'solution': "The number of ways to roll exactly 2 6's is $\\binom{5}{2}5^3$, since there are $\\binom{5}{2}$ choices for which of the two dice are 6, and there are 5 choices for each of the other 3 dice. Similarly, the number of ways to roll exactly 1 6 is $\\binom{5}{1}5^4$, and the number of ways to roll no 6's is $\\binom{5}{0}5^5$. So the probability is \\[\\frac{\\binom{5}{2}5^3+\\binom{5}{1}5^4+\\binom{5}{0}5^5}{6^5}=\\boxed{\\frac{625}{648}}.\\]"}
        self.sample_data2 = {'problem': 'When counting from $3$ to $201$, $53$ is the $51^\\mathrm{st}$ number counted. When counting backwards from $201$ to $3$, $53$ is the $n^\\mathrm{th}$ number counted. What is $n$?', 'level': 'Level 2', 'type': 'Counting & Probability', 'solution': 'Note that $n$ is equal to the number of integers between $53$ and $201$, inclusive. Thus, $n=201-53+1=\\boxed{149}$.'}

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        train_file = os.path.join(self.temp_dir, 'MATH', 'train', 'Counting & Probability', 'sample1.json')
        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        save_json(self.sample_data1, train_file, type='json')
        test_file = os.path.join(self.temp_dir, 'MATH', 'test', 'Counting & Probability', 'sample1.json')
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        save_json(self.sample_data2, test_file, type='json')

    @patch('evoagentx.benchmark.math_benchmark.download_raw_math_data')
    def test_load_data(self, mock_download):
        self.create_test_files()
        benchmark = MATH(path=self.temp_dir)
        self.assertEqual(len(benchmark.get_train_data()), 1)
        self.assertEqual(len(benchmark.get_dev_data()), 0)
        self.assertEqual(len(benchmark.get_test_data()), 1)
        self.assertEqual(mock_download.call_count, 0)

    def test_get_label(self):
        self.create_test_files()
        benchmark = MATH(path=self.temp_dir, mode='train')
        example = benchmark.get_train_data()[0]
        self.assertEqual(benchmark.get_label(example), self.sample_data1['solution'])
        self.assertEqual(benchmark.get_id(example), 'train-1')

    def test_extract_answer(self):
        self.create_test_files()
        benchmark = MATH(path=self.temp_dir, mode='train')
        example = benchmark.get_train_data()[0]
        self.assertEqual(benchmark.extract_answer(example['solution']), '\\frac{625}{648}')

    def test_evaluate(self):
        self.create_test_files()
        benchmark = MATH(path=self.temp_dir, mode='train')
        example = benchmark.get_train_data()[0]
        prediction = benchmark.extract_answer(example['solution'])
        self.assertEqual(str(prediction), str('\\frac{625}{648}'))
        self.assertTrue(benchmark.math_equal(prediction, '\\frac{625}{648}'))
        self.assertFalse(benchmark.math_equal(prediction, '\\frac{625}{649}'))
        self.assertFalse(benchmark.is_digit(prediction))
        self.assertFalse(benchmark.is_digit('\\frac{625}{648}'))
        self.assertTrue(benchmark.symbolic_equal(prediction, '\\frac{625}{648}'))
        self.assertFalse(benchmark.symbolic_equal(prediction, '\\frac{625}{649}'))
        self.assertEqual(benchmark.evaluate(example['solution'], '\\frac{625}{648}'), {'solve_rate': 1.0})
        self.assertEqual(benchmark.evaluate(example['solution'], '\\frac{625}{649}'), {'solve_rate': 0.0})

class TestGSM8K(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [{'question': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", 'answer': 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18'}, {'question': 'A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?', 'answer': 'It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3'}]

    def tearDown(self):
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def create_test_file(self, filename, data):
        filepath = os.path.join(self.temp_dir, filename)
        save_json(data=data, path=filepath, type='jsonl')
        return filepath

    @patch('evoagentx.benchmark.gsm8k.download_raw_gsm8k_data')
    def test_load_data(self, mock_download):
        self.create_test_file('train.jsonl', self.sample_data)
        self.create_test_file('test.jsonl', self.sample_data)
        benchmark = GSM8K(path=self.temp_dir)
        self.assertEqual(len(benchmark.get_train_data()), 2)
        self.assertEqual(len(benchmark.get_dev_data()), 0)
        self.assertEqual(len(benchmark.get_test_data()), 2)
        self.assertEqual(mock_download.call_count, 0)

    def test_get_label(self):
        self.create_test_file('train.jsonl', self.sample_data)
        benchmark = GSM8K(path=self.temp_dir, mode='train')
        example = benchmark.get_train_data()[0]
        self.assertEqual(benchmark.get_label(example), self.sample_data[0]['answer'])
        self.assertEqual(benchmark.get_id(example), 'train-1')

    def test_extract_last_number(self):
        self.create_test_file('train.jsonl', self.sample_data)
        benchmark = GSM8K(path=self.temp_dir)
        self.assertEqual(benchmark.extract_last_number(benchmark.get_train_data()[0]['answer']), 18)
        self.assertEqual(benchmark.extract_last_number(benchmark.get_train_data()[1]['answer']), 3)
        self.assertEqual(benchmark.extract_last_number('The answer is123.45'), 123.45)
        self.assertEqual(benchmark.extract_last_number('The answer is: xxx123.45'), 123.45)
        self.assertEqual(benchmark.extract_last_number('The answer is:\n123.45'), 123.45)
        self.assertEqual(benchmark.extract_last_number('The answer is:\n #### 123.45'), 123.45)

    def test_evaluate(self):
        self.create_test_file('train.jsonl', self.sample_data)
        benchmark = GSM8K(path=self.temp_dir, mode='train')
        result = benchmark.evaluate(prediction='18', label=self.sample_data[0]['answer'])
        self.assertEqual(result['solve_rate'], 1.0)
        result = benchmark.evaluate(prediction='reasoning process, ####18', label=self.sample_data[0]['answer'])
        self.assertEqual(result['solve_rate'], 1.0)
        result = benchmark.evaluate(prediction='wrong answer 111', label=self.sample_data[0]['answer'])
        self.assertEqual(result['solve_rate'], 0.0)


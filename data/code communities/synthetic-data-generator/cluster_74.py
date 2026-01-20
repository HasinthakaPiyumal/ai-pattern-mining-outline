# Cluster 74

class SingleTableGPTModel(LLMBaseModel):
    """
    This is a synthetic data generation model powered by OpenAI GPT, a state-of-the-art language model. This model is based on groundbreaking research presented in the ICLR paper titled "Language Models are Realistic Tabular Data Generators".

    Our model harnesses the power of GPT to generate synthetic tabular data that closely resembles real-world datasets. By utilizing the advanced capabilities of GPT, we aim to provide a reliable and efficient solution for generating simulated data that can be used for various purposes, such as testing, training, and analysis.

    With this synthetic data generation model, users can easily generate diverse and realistic tabular datasets, mimicking the characteristics and patterns found in real data.
    """
    openai_API_key = ''
    '\n    The API key required to access the OpenAI GPT model. Please provide your own API key for authentication.\n    '
    openai_API_url = 'https://api.openai.com/v1/'
    '\n    The URL endpoint for the OpenAI GPT API. Please specify the appropriate URL for accessing the API.\n    '
    max_tokens = 4000
    '\n    The maximum number of tokens allowed in the generated response. This parameter helps in limiting the length of the output text.\n    '
    temperature = 0.1
    '\n    A parameter that controls the randomness of the generated text. Lower values like 0.1 make the output more focused and deterministic, while higher values like 1.0 introduce more randomness.\n    '
    timeout = 90
    '\n    The maximum time (in seconds) to wait for a response from the OpenAI GPT API. If the response is not received within this time, the request will be timed out.\n    '
    gpt_model = 'gpt-3.5-turbo'
    '\n    The specific GPT model to be used for generating text. The default model is "gpt-3.5-turbo", which is known for its high performance and versatility.\n    '
    query_batch = 30
    '\n    This parameter is the number of samples submitted to GPT each time and the number of returned samples.\n\n    This size has a certain relationship with the max_token parameter.\n\n    We do not recommend setting too large a value, as this may cause potential problems or errors.\n    '
    _sample_lines = []
    '\n    A list to store the sample lines of generated data.\n    '
    _result_list = []
    '\n    A list to store the generated data samples.\n    '

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the class instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._get_openai_setting_from_env()

    def check(self):
        """
        Performs various checks.

        Raises:
            SynthesizerInitError: If data access type is not specified or if duplicate data access type is found.
        """
        self._check_openAI_setting()
        self._set_openAI()
        self._check_access_type()

    def set_openAI_settings(self, API_url='https://api.openai.com/v1/', API_key=''):
        """
        Sets the OpenAI settings.

        Args:
            API_url (str): The OpenAI API URL. Defaults to "https://api.openai.com/v1/".
            API_key (str): The OpenAI API key. Defaults to an empty string.
        """
        self.openai_API_url = API_url
        self.openai_API_key = API_key
        self._set_openAI()

    def _set_openAI(self):
        """
        Sets the OpenAI API key and base URL.
        """
        openai.api_key = self.openai_API_key
        openai.base_url = self.openai_API_url

    def _check_openAI_setting(self):
        """
        Checks if the OpenAI settings are properly initialized.

        Raises:
            InitializationError: If openai_API_url or openai_API_key is not found.
        """
        if not self.openai_API_url:
            raise InitializationError('openai_API_url NOT found.')
        if not self.openai_API_key:
            raise InitializationError('openai_API_key NOT found.')
        logger.debug('OpenAI setting check passed.')

    def _get_openai_setting_from_env(self):
        """
        Retrieves OpenAI settings from environment variables.
        """
        if os.getenv('OPENAI_KEY'):
            self.openai_API_key = os.getenv('OPENAI_KEY')
            logger.debug('Get OPENAI_KEY from ENV.')
        if os.getenv('OPENAI_URL'):
            self.openai_API_url = os.getenv('OPENAI_URL')
            logger.debug('Get OPENAI_URL from ENV.')

    def openai_client(self):
        """
        Generate a openai request client.
        """
        return openai.OpenAI(api_key=self.openai_API_key, base_url=self.openai_API_url)

    def ask_gpt(self, question, model=None):
        """
        Sends a question to the GPT model.

        Args:
            question (str): The question to ask.
            model (str): The GPT model to use. Defaults to None.

        Returns:
            str: The response from the GPT model.

        Raises:
            SynthesizerInitError: If the check method fails.
        """
        self.check()
        if model:
            model = model
        else:
            model = self.gpt_model
        client = self.openai_client()
        logger.info(f'Ask GPT with temperature = {self.temperature}.')
        response = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': question}], temperature=self.temperature, max_tokens=self.max_tokens, timeout=self.timeout)
        logger.info('Ask GPT Finished.')
        self._responses.append(response)
        return response.choices[0].message.content

    def fit(self, raw_data: pd.DataFrame | DataLoader=None, metadata: Metadata=None, *args, **kwargs):
        """
        Fits this model to the provided data.
        Please note that no actual algorithmic training is excuted here.

        Args:
            raw_data (pd.DataFrame | DataLoader): The raw data to fit the model to. It can be either a pandas DataFrame or a DataLoader object.
            metadata (Metadata): The metadata associated with the raw data.

        Returns:
            None

        Raises:
            InitializationError: If neither raw_data nor metadata is provided.
        """
        if raw_data is not None and type(raw_data) in [pd.DataFrame, DataLoader]:
            if metadata:
                self._metadata = metadata
            self._fit_with_data(raw_data)
            return
        if type(raw_data) is Metadata:
            self._fit_with_metadata(raw_data)
            return
        if metadata is not None and type(metadata) is Metadata:
            self._fit_with_metadata(metadata)
            return
        raise InitializationError('Ple1ase pass at least one valid parameter, train_data or metadata')

    def _fit_with_metadata(self, metadata):
        """
        Fit the model using metadata.

        Args:
            metadata: Metadata object.

        Returns:
            None
        """
        logger.info('Fitting model with metadata...')
        self.use_metadata = True
        self._metadata = metadata
        self.columns = list(metadata.column_list)
        logger.info('Fitting model with metadata... Finished.')

    def _fit_with_data(self, train_data):
        """
        Fit the model using data.

        Args:
            train_data: Training data.

        Returns:
            None
        """
        logger.info('Fitting model with raw data...')
        self.use_raw_data = True
        self.use_dataloader = False
        if type(train_data) is DataLoader:
            self.columns = list(train_data.columns())
            train_data = train_data.load_all()
        if not self.columns:
            self.columns = list(train_data.columns)
        if not self._metadata:
            self._metadata = Metadata.from_dataframe(train_data)
        sample_lines = []
        for _, row in train_data.iterrows():
            each_line = ''
            shuffled_columns = copy(self.columns)
            random.shuffle(shuffled_columns)
            for column in shuffled_columns:
                value = str(row[column])
                each_line += f'{column} is {value}, '
            each_line = each_line[:-2]
            each_line += '\n'
            sample_lines.append(each_line)
        self._sample_lines = sample_lines
        logger.info('Fitting model with raw data... Finished.')

    @staticmethod
    def _select_random_elements(input_list, cnt):
        """
        This function selects a random sample of elements from the input list.

        Args:
            input_list (list): The list from which elements will be selected.
            cnt (int): The number of elements to be selected.

        Returns:
            list: A list of randomly selected elements from the input list.

        Raises:
            ValueError: If cnt is greater than the length of the input list.
        """
        if cnt > len(input_list):
            raise ValueError('cnt should not be greater than the length of the list')
        return random.sample(input_list, cnt)

    def _form_message_with_data(self, sample_list, current_cnt):
        """
        This function forms a message with data.

        Args:
            sample_list (list): A list of samples.
            current_cnt (int): The current count of samples.

        Returns:
            str: The formed message with data.
        """
        sample_str = ''
        for i in range(current_cnt):
            each_sample = sample_list[i]
            each_str = f'sample {i}: ' + each_sample + '\n'
            sample_str += each_str
        message = self.prompts['message_prefix'] + sample_str
        message = message + self._form_dataset_description()
        message = message + self._form_message_with_offtable_features()
        message = message + f'Please note that the generated table has total {len(self.columns) + len(self.off_table_features)} columns of the generated data, the column names are {self.columns + self.off_table_features}, every column should not be missed when generating the data. \n'
        message = message + self.prompts['message_suffix'] + str(current_cnt) + '.'
        self._message_list.append(message)
        logger.debug('Message Generated.')
        return message

    def extract_samples_from_response(self, response_content):
        """
        Extracts samples from the response content.

        Args:
            response_content (dict): The response content as a dictionary.

        Returns:
            list: A list of extracted samples.
        """

        def dict_to_list(input_dict, header):
            """
            Converts a dictionary to a list based on the given header.

            Args:
                input_dict (dict): The input dictionary.
                header (list): The list of keys to extract from the dictionary.

            Returns:
                list: A list of values extracted from the dictionary based on the header.
            """
            res = []
            for each_col in header:
                each_value = input_dict.get(each_col, None)
                res.append(each_value)
            return res
        logger.info('Extracting samples from response ...')
        header = self.columns + self.off_table_features
        features = []
        for line in response_content.split('\n'):
            feature = {}
            for field in header:
                pattern = '\\b' + field + '\\s*(?:is|=)\\s*([^,\\n]+)'
                match = re.search(pattern, line)
                if match:
                    feature[field] = match.group(1).strip()
            if feature:
                features.append(dict_to_list(feature, header))
        logger.info(f'Extracting samples from response ... Finished, {len(features)} extracted.')
        return features

    def sample(self, count=50, dataset_desp='', *args, **kwargs):
        """
        This function samples data from either raw data or metadata based on the given parameters.

        Args:
            count (int): The number of samples to be generated. Default is 50.
            dataset_desp (str): The description of the dataset. Default is an empty string.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            res: The sampled data.
        """
        logger.info('Sampling use GPT model ...')
        self.dataset_description = dataset_desp
        if self.use_raw_data:
            res = self._sample_with_data(count, *args, **kwargs)
        elif self.use_metadata:
            res = self._sample_with_metadata(count, *args, **kwargs)
        logger.info('Sampling use GPT model ... Finished.')
        return res

    def _form_message_with_metadata(self, current_cnt):
        """
        This function forms a message with metadata for table data generation task.

        Args:
            current_cnt (int): The current count of the message.

        Returns:
            str: The formed message with metadata.
        """
        message = ''
        message = message + self.prompts['message_prefix']
        message = message + self._form_dataset_description()
        message = message + 'This table data generation task will only have metadata and no data samples. The header (columns infomation) of the tabular data is: '
        message = message + str(self.columns) + '. \n'
        message = message + self._form_message_with_offtable_features()
        message = message + f'Note that the generated table has total {len(self.columns) + len(self.off_table_features)} columns, the column names are {self.columns + self.off_table_features}, every column should NOT be missed in generated data.\n'
        message = message + self.prompts['message_suffix'] + str(current_cnt) + '.'
        self._message_list.append(message)
        return message

    def _sample_with_metadata(self, count, *args, **kwargs):
        """
        This method samples data with metadata.

        Args:
            count (int): The number of samples to be generated.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            int: The input count.

        """
        logger.info('Sampling with metadata.')
        result = []
        remaining_cnt = count
        while remaining_cnt > 0:
            if remaining_cnt - self.query_batch >= 0:
                current_cnt = self.query_batch
            else:
                current_cnt = remaining_cnt
            message = self._form_message_with_metadata(current_cnt)
            response = self.ask_gpt(message)
            generated_batch = self.extract_samples_from_response(response)
            result += generated_batch
            remaining_cnt = remaining_cnt - current_cnt
        self._result_list.append(result)
        final_columns = self.columns + self.off_table_features
        return pd.DataFrame(result, columns=final_columns)

    def _sample_with_data(self, count, *args, **kwargs):
        """
        This function samples data with a given count and returns a DataFrame with the sampled data.

        Args:
            count (int): The number of data samples to be generated.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled data.

        """
        logger.info('Sampling with raw_data.')
        result = []
        remaining_cnt = count
        while remaining_cnt > 0:
            if remaining_cnt - self.query_batch >= 0:
                current_cnt = self.query_batch
            else:
                current_cnt = remaining_cnt
            sample_list = self._select_random_elements(self._sample_lines, current_cnt)
            message = self._form_message_with_data(sample_list, current_cnt)
            response = self.ask_gpt(message)
            generated_batch = self.extract_samples_from_response(response)
            result += generated_batch
            remaining_cnt = remaining_cnt - current_cnt
        self._result_list.append(result)
        final_columns = self.columns + self.off_table_features
        return pd.DataFrame(result, columns=final_columns)

@pytest.fixture
def demo_single_table_data_pos_neg_metadata(demo_single_table_data_pos_neg):
    metadata = Metadata.from_dataframe(demo_single_table_data_pos_neg.copy(), check=True)
    metadata.categorical_encoder = {'cat_onehot': 'onehot', 'cat_label': 'label', 'cat_freq': 'frequency'}
    metadata.datetime_format = {'cat_date': '%Y-%m-%d'}
    metadata.categorical_threshold = {99: 'frequency', 199: 'label'}
    yield metadata

def test_email_generator(chn_personal_test_df: pd.DataFrame):
    assert 'email' in chn_personal_test_df.columns
    metadata_df = Metadata.from_dataframe(chn_personal_test_df)
    email_generator = EmailGenerator()
    assert not email_generator.fitted
    email_generator.fit(metadata_df)
    assert email_generator.fitted
    assert email_generator.email_columns_list == ['email']
    converted_df = email_generator.convert(chn_personal_test_df)
    assert len(converted_df) == len(chn_personal_test_df)
    assert converted_df.shape[1] != chn_personal_test_df.shape[1]
    assert converted_df.shape[1] == chn_personal_test_df.shape[1] - len(email_generator.email_columns_list)
    assert 'email' not in converted_df.columns
    reverse_converted_df = email_generator.reverse_convert(converted_df)
    assert len(reverse_converted_df) == len(chn_personal_test_df)
    assert 'email' in reverse_converted_df.columns
    for each_value in chn_personal_test_df['email'].values:
        assert EmailCheckModel(email=each_value)

def test_chn_pii_generator(chn_personal_test_df: pd.DataFrame):
    assert 'chn_name' in chn_personal_test_df.columns
    assert 'mobile_phone_no' in chn_personal_test_df.columns
    assert 'ssn_sfz' in chn_personal_test_df.columns
    assert 'company_name' in chn_personal_test_df.columns
    metadata_df = Metadata.from_dataframe(chn_personal_test_df)
    pii_generator = ChnPiiGenerator()
    assert not pii_generator.fitted
    pii_generator.fit(metadata_df)
    assert pii_generator.fitted
    assert pii_generator.chn_name_columns_list == ['chn_name']
    assert pii_generator.chn_phone_columns_list == ['mobile_phone_no']
    assert pii_generator.chn_id_columns_list == ['ssn_sfz']
    assert pii_generator.chn_company_name_list == ['company_name']
    converted_df = pii_generator.convert(chn_personal_test_df)
    assert len(converted_df) == len(chn_personal_test_df)
    assert converted_df.shape[1] != chn_personal_test_df.shape[1]
    assert converted_df.shape[1] == chn_personal_test_df.shape[1] - len(pii_generator.chn_pii_columns)
    assert 'chn_name' not in converted_df.columns
    assert 'mobile_phone_no' not in converted_df.columns
    assert 'ssn_sfz' not in converted_df.columns
    assert 'company_name' not in converted_df.columns
    reverse_converted_df = pii_generator.reverse_convert(converted_df)
    assert len(reverse_converted_df) == len(chn_personal_test_df)
    assert 'chn_name' in reverse_converted_df.columns
    assert 'mobile_phone_no' in reverse_converted_df.columns
    assert 'ssn_sfz' in reverse_converted_df.columns
    assert 'company_name' in reverse_converted_df.columns
    for each_value in chn_personal_test_df['ssn_sfz'].values:
        assert len(each_value) == 18
        pattern = '^\\d{17}[0-9X]$'
        assert bool(re.match(pattern, each_value))
    for each_value in chn_personal_test_df['chn_name'].values:
        pattern = '^[\\u4e00-\\u9fa5]{2,5}$'
        assert len(each_value) >= 2 and len(each_value) <= 5
        assert bool(re.match(pattern, each_value))
    for each_value in chn_personal_test_df['mobile_phone_no'].values:
        assert each_value.startswith('1')
        assert len(each_value) == 11
        pattern = '^1[3-9]\\d{9}$'
        assert bool(re.match(pattern, each_value))
    for each_value in chn_personal_test_df['company_name'].values:
        pattern = '.*?公司.*?'
        assert bool(re.match(pattern, each_value))

def test_positive_negative_filter(pos_neg_test_df: pd.DataFrame):
    metadata_df = Metadata.from_dataframe(pos_neg_test_df)
    pos_neg_filter = PositiveNegativeFilter()
    assert not pos_neg_filter.fitted
    pos_neg_filter.fit(metadata_df)
    assert pos_neg_filter.fitted
    assert pos_neg_filter.positive_columns == {'int_id', 'pos_int', 'pos_float'}
    assert pos_neg_filter.negative_columns == {'neg_int', 'neg_float'}
    converted_df = pos_neg_filter.convert(pos_neg_test_df)
    assert converted_df.shape == pos_neg_test_df.shape
    assert (converted_df['pos_int'] >= 0).all()
    assert (converted_df['pos_float'] >= 0).all()
    assert (converted_df['neg_int'] <= 0).all()
    assert (converted_df['neg_float'] <= 0).all()
    reverse_converted_df = pos_neg_filter.reverse_convert(converted_df)
    assert reverse_converted_df.shape[1] == converted_df.shape[1]
    assert (reverse_converted_df['pos_int'] >= 0).all()
    assert (reverse_converted_df['pos_float'] >= 0).all()
    assert (reverse_converted_df['neg_int'] <= 0).all()
    assert (reverse_converted_df['neg_float'] <= 0).all()
    pd.testing.assert_series_equal(pos_neg_test_df['mixed_int'], reverse_converted_df['mixed_int'])
    pd.testing.assert_series_equal(pos_neg_test_df['mixed_float'], reverse_converted_df['mixed_float'])
    assert reverse_converted_df.shape[0] <= pos_neg_test_df.shape[0]

def test_specific_combination_transformer(train_data, test_data):
    transformer = SpecificCombinationTransformer()
    metadata = Metadata.from_dataframe(train_data)
    combinations = {('price_usd', 'price_cny', 'price_eur'), ('size_cm', 'size_inch', 'size_m')}
    metadata.update({'specific_combinations': combinations})
    transformer.fit(metadata=metadata, tabular_data=train_data)
    result = transformer.reverse_convert(test_data)
    for cols in combinations:
        result_rows = result[list(cols)].values.tolist()
        train_rows = train_data[list(cols)].values.tolist()
        assert all((row in train_rows for row in result_rows)), f'Combination {cols} contains some invalid value in original train data.'

def test_numeric_transformer_fit_test_df(df_data: pd.DataFrame):
    """ """
    metadata_df = Metadata.from_dataframe(df_data)
    transformer = NumericValueTransformer()
    transformer.fit(metadata_df, df_data)
    assert transformer.int_columns == {'int_random', 'int_id'}
    assert transformer.float_columns == {'float_random'}

def test_numeric_transformer_convert_test_df(df_data: pd.DataFrame):
    """ """
    metadata_df = Metadata.from_dataframe(df_data)
    transformer = NumericValueTransformer()
    transformer.fit(metadata_df, df_data)
    converted_df = transformer.convert(df_data)
    numerical_columns = list(transformer.int_columns) + list(transformer.float_columns)
    converted_status = calculate_mean_and_variance(converted_df, numerical_columns)
    assert type(converted_df) == pd.DataFrame
    assert converted_df.shape == df_data.shape
    assert np.isclose(converted_status['int_id']['mean'], 0.0)
    assert np.isclose(converted_status['int_random']['mean'], 0.0)
    assert np.isclose(converted_status['float_random']['mean'], 0.0)
    assert np.isclose(converted_status['int_id']['variance'], 1, atol=0.001)
    assert np.isclose(converted_status['int_random']['variance'], 1, atol=0.001)
    assert np.isclose(converted_status['float_random']['variance'], 1, atol=0.001)

def calculate_mean_and_variance(df, numeric_df):
    if not isinstance(numeric_df, list):
        raise ValueError('numeric_df should be a list of column names.')
    for col in numeric_df:
        if col not in df.columns:
            raise ValueError(f'Column {col} does not exist in the DataFrame.')
    stats = {}
    for col in numeric_df:
        mean = df[col].mean()
        variance = df[col].var()
        stats[col] = {'mean': mean, 'variance': variance}
    return stats

def test_numeric_transformer_reverse_convert_test_df(df_data: pd.DataFrame):
    """ """
    transformer = NumericValueTransformer()
    transformer.fit(Metadata.from_dataframe(df_data), df_data)
    numerical_columns = list(transformer.int_columns) + list(transformer.float_columns)
    converted_df = transformer.convert(df_data)
    reverse_converted_df = transformer.reverse_convert(converted_df)
    reverse_converted_status = calculate_mean_and_variance(reverse_converted_df, numerical_columns)
    original_status = calculate_mean_and_variance(df_data, numerical_columns)
    assert type(reverse_converted_df) == pd.DataFrame
    assert reverse_converted_df.shape == df_data.shape
    assert np.isclose(reverse_converted_status['int_id']['mean'], original_status['int_id']['mean'])
    assert np.isclose(reverse_converted_status['int_random']['mean'], original_status['int_random']['mean'])
    assert np.isclose(reverse_converted_status['float_random']['mean'], original_status['float_random']['mean'])
    assert np.isclose(reverse_converted_status['int_id']['variance'], original_status['int_id']['variance'])
    assert np.isclose(reverse_converted_status['int_random']['variance'], original_status['int_random']['variance'])
    assert np.isclose(reverse_converted_status['float_random']['variance'], original_status['float_random']['variance'])

def test_empty_handling_test_df(test_empty_data: pd.DataFrame):
    """
    Test the handling of empty columns in a DataFrame.
    This function tests the behavior of a DataFrame when it contains empty columns.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
    test_empty_data (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If the DataFrame does not handle empty columns as expected.
    """
    metadata = Metadata.from_dataframe(test_empty_data)
    empty_transformer = EmptyTransformer()
    assert empty_transformer.fitted is False
    empty_transformer.fit(metadata)
    assert empty_transformer.fitted
    assert sorted(empty_transformer.empty_columns) == ['age', 'fnlwgt']
    transformed_df = empty_transformer.convert(test_empty_data)
    processed_metadata = Metadata.from_dataframe(transformed_df)
    assert not processed_metadata.get('empty_columns')
    reverse_converted_df = empty_transformer.reverse_convert(transformed_df)
    reverse_converted_metadata = Metadata.from_dataframe(reverse_converted_df)
    assert reverse_converted_metadata.get('empty_columns') == {'age', 'fnlwgt'}

def test_fixed_combination_handling_test_df(test_fixed_combination_data: pd.DataFrame):
    """
    Test the handling of fixed combination columns in a DataFrame.
    This function tests the behavior of a DataFrame when it contains fixed combination columns.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
    test_fixed_combination_data (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If the DataFrame does not handle fixed combination columns as expected.
    """
    metadata = Metadata.from_dataframe(test_fixed_combination_data)
    fixed_combinations = metadata.get('fixed_combinations')
    assert fixed_combinations == {'A': {'E', 'D', 'B'}, 'B': {'A', 'E', 'D'}, 'D': {'A', 'E', 'B'}, 'E': {'A', 'D', 'B'}, 'categorical_one': {'categorical_two'}}
    fixed_combination_transformer = FixedCombinationTransformer()
    assert fixed_combination_transformer.fitted is False
    fixed_combination_transformer.fit(metadata)
    assert fixed_combination_transformer.fitted
    assert fixed_combination_transformer.fixed_combinations == {'A': {'E', 'D', 'B'}, 'B': {'A', 'E', 'D'}, 'D': {'A', 'E', 'B'}, 'E': {'A', 'D', 'B'}, 'categorical_one': {'categorical_two'}}
    transformed_df = fixed_combination_transformer.convert(test_fixed_combination_data)
    for column in test_fixed_combination_data.columns:
        assert column in transformed_df.columns, f'Column {column} should be retained in the transformed data.'
    assert transformed_df.shape == test_fixed_combination_data.shape

def test_categorical_fixed_combinations(test_fixed_combination_data):
    """Test the fixed combination relationship of categorical variables"""
    metadata = Metadata.from_dataframe(test_fixed_combination_data)
    transformer = FixedCombinationTransformer()
    transformer.fit(metadata)
    assert 'categorical_one' in transformer.fixed_combinations
    assert 'categorical_two' in transformer.fixed_combinations['categorical_one']
    transformed_df = transformer.convert(test_fixed_combination_data)
    assert all(transformed_df['categorical_one'].map(dict(zip(test_fixed_combination_data['categorical_one'], test_fixed_combination_data['categorical_two']))) == transformed_df['categorical_two'])

def test_numeric_transformer_fit_test_df(df_data: pd.DataFrame, df_data_processed: pd.DataFrame):
    """
    Test the functionality of the ColumnOrderTransformer class.

    This function tests the following:
    1. The correctness of the input dataframes' columns and shapes.
    2. The correctness of the metadata extraction from the input dataframe.
    3. The correctness of the fitting of the ColumnOrderTransformer.
    4. The correctness of the conversion of the input dataframe using the ColumnOrderTransformer.
    5. The correctness of the reverse conversion of the processed dataframe using the ColumnOrderTransformer.

    Parameters:
    df_data (pd.DataFrame): The input dataframe to be transformed.
    df_data_processed (pd.DataFrame): The processed dataframe to be reversely transformed.

    Returns:
    None
    """
    assert df_data.columns.to_list() == ['int_id', 'str_id', 'int_random', 'bool_random', 'float_random']
    assert df_data_processed.columns.to_list() == ['int_random', 'int_id', 'float_random_2', 'bool_random', 'float_random', 'bool_random_2', 'str_id']
    assert df_data.shape == (100, 5)
    assert df_data_processed.shape == (100, 7)
    metadata_df = Metadata.from_dataframe(df_data)
    transformer = ColumnOrderTransformer()
    transformer.fit(metadata_df)
    assert transformer.column_list == ['int_id', 'str_id', 'int_random', 'bool_random', 'float_random']
    transformed_df = transformer.convert(df_data)
    assert transformed_df.columns.to_list() == df_data.columns.to_list()
    assert transformed_df.shape == (100, 5)
    assert df_data.equals(transformed_df)
    convert_transformed_df = transformer.reverse_convert(df_data_processed)
    assert df_data.columns.to_list() == convert_transformed_df.columns.to_list()
    assert convert_transformed_df.shape == (100, 5)
    assert convert_transformed_df.columns.to_list() == transformer.column_list

@pytest.mark.skip(reason='success in local, failed in GitHub Action')
def test_outlier_handling_test_df(outlier_test_df: pd.DataFrame):
    """
    Test the handling of outliers in a DataFrame.
    This function tests the behavior of a DataFrame when it contains outliers.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
        outlier_test_df (pd.DataFrame): The DataFrame to test.

    Returns:
        None

    Raises:
        AssertionError: If the DataFrame does not handle outliers as expected.
    """
    assert 'not_number_outlier' in outlier_test_df['int_random'].to_list()
    assert 'not_number_outlier' in outlier_test_df['float_random'].to_list()
    outlier_transformer = OutlierTransformer()
    assert outlier_transformer.fitted is False
    metadata_outlier = Metadata.from_dataframe(outlier_test_df)
    metadata_outlier.column_list = ['int_id', 'str_id', 'int_random', 'float_random']
    metadata_outlier.int_columns = set(['int_id', 'int_random'])
    metadata_outlier.float_columns = set(['float_random'])
    outlier_transformer.fit(metadata=metadata_outlier)
    assert outlier_transformer.fitted
    transformed_df = outlier_transformer.convert(outlier_test_df)
    assert not 'not_number_outlier' in transformed_df['int_random'].to_list()
    assert not 'not_number_outlier' in transformed_df['float_random'].to_list()
    assert 0 in transformed_df['int_random'].to_list()
    assert 0.0 in transformed_df['float_random'].to_list()

@pytest.mark.skip(reason='success in local, failed in GitHub Action')
def test_nan_handling_test_df(nan_test_df: pd.DataFrame):
    """
    Test the handling of NaN values in a DataFrame.
    This function tests the behavior of a DataFrame when it contains NaN values.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
    nan_test_df (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If the DataFrame does not handle NaN values as expected.
    """
    assert has_nan(nan_test_df), 'NaN values were not removed from the DataFrame.'
    nan_transformer = NonValueTransformer()
    assert nan_transformer.fitted is False
    nan_csv_metadata = Metadata.from_dataframe(nan_test_df)
    nan_csv_metadata.column_list = ['int_id', 'str_id', 'int_random', 'bool_random']
    nan_transformer.fit(nan_csv_metadata)
    assert nan_transformer.fitted
    transformed_df = nan_transformer.convert(nan_test_df)
    assert not has_nan(transformed_df)

def has_nan(df):
    """
    This function checks if there are any NaN values in the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to check for NaN values.

    Returns:
    bool: True if there is any NaN value in the DataFrame, False otherwise.
    """
    return df.isnull().values.any()

def test_discrete_transformer_fit_test_df(df_data: pd.DataFrame):
    """
    Test the fit and convert methods of the DiscreteTransformer class.

    This function tests the following:
    1. The fit method of the DiscreteTransformer class.
    2. The convert method of the DiscreteTransformer class.
    3. The reverse_convert method of the DiscreteTransformer class.
    4. The equality of the original dataframe and the reversely converted dataframe.

    Parameters:
    df_data (pd.DataFrame): The input dataframe to be tested.

    Returns:
    None
    """
    metadata_df = Metadata.from_dataframe(df_data)
    order_transformer = ColumnOrderTransformer()
    order_transformer.fit(metadata_df)
    transformer = DiscreteTransformer()
    assert not transformer.fitted
    transformer.fit(metadata_df, df_data)
    assert transformer.fitted
    assert transformer.discrete_columns == {'discrete_val'}
    converted_df = transformer.convert(df_data)
    assert isinstance(converted_df, pd.DataFrame)
    assert is_an_integer_list(converted_df['discrete_val_a'].to_list())
    assert is_an_integer_list(converted_df['discrete_val_b'].to_list())
    assert is_an_integer_list(converted_df['discrete_val_c'].to_list())
    reverse_converted_df = transformer.reverse_convert(converted_df)
    reverse_converted_df = order_transformer.reverse_convert(reverse_converted_df)
    assert isinstance(reverse_converted_df, pd.DataFrame)
    assert is_a_string_list(reverse_converted_df['discrete_val'].to_list())
    assert reverse_converted_df.eq(df_data).all().all()

def is_an_integer_list(lst):
    """
    Check if all elements in the list are integers or floats that are also integers.

    Parameters:
    lst (list): The list to be checked.

    Returns:
    bool: True if all elements are integers or floats that are also integers, False otherwise.
    """
    return all((isinstance(i, int) or (isinstance(i, float) and i.is_integer()) for i in lst))

def is_a_string_list(lst):
    """
    Check if all items in a list are strings.

    Parameters:
    lst (list): The list to check.

    Returns:
    bool: True if all items in the list are strings, False otherwise.
    """
    return all((isinstance(item, str) for item in lst))

def test_const_handling_test_df(test_const_data: pd.DataFrame):
    """
    Test the handling of const columns in a DataFrame.
    This function tests the behavior of a DataFrame when it contains const columns.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
    test_const_data (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If the DataFrame does not handle const columns as expected.
    """
    metadata = Metadata.from_dataframe(test_const_data)
    const_transformer = ConstValueTransformer()
    assert const_transformer.fitted is False
    const_transformer.fit(metadata)
    assert const_transformer.fitted
    assert sorted(const_transformer.const_columns) == ['age', 'fnlwgt', 'workclass']
    transformed_df = const_transformer.convert(test_const_data)
    assert 'age' not in transformed_df.columns
    assert 'fnlwgt' not in transformed_df.columns
    assert 'workclass' not in transformed_df.columns
    reverse_converted_df = const_transformer.reverse_convert(transformed_df)
    assert 'age' in reverse_converted_df.columns
    assert 'fnlwgt' in reverse_converted_df.columns
    assert 'workclass' in reverse_converted_df.columns
    assert reverse_converted_df['age'][0] == 100
    assert reverse_converted_df['fnlwgt'][0] == 1.41421
    assert reverse_converted_df['workclass'][0] == 'President'
    assert len(reverse_converted_df['age'].unique()) == 1
    assert len(reverse_converted_df['fnlwgt'].unique()) == 1
    assert len(reverse_converted_df['workclass'].unique()) == 1

@pytest.mark.skip(reason='success in local, failed in GitHub Action')
def test_int_formatter_fit_test_df():
    """
    Test the functionality of the IntValueFormatter class.

    This function tests the following:
    1. The fit method of the IntValueFormatter class.
    2. The addition of a new column to the formatter.
    3. The reverse conversion of the DataFrame.
    4. The checking of integer values in the DataFrame columns.

    Parameters:
    df_data (pd.DataFrame): The DataFrame to be tested.

    Returns:
    None

    Raises:
    AssertionError: If any of the assertions fail.
    """
    df = int_formatter_df()
    metadata_df = Metadata.from_dataframe(df)
    formatter = IntValueFormatter()
    formatter.fit(metadata_df)
    metadata_df.column_list = ['int_id', 'str_id', 'int_random', 'float_random']
    assert sorted(metadata_df.column_list) == sorted(['int_id', 'str_id', 'int_random', 'float_random'])
    assert 'int_random' in formatter.int_columns
    assert 'int_id' in formatter.int_columns
    reverse_df = formatter.reverse_convert(df)
    assert is_an_integer_list(reverse_df['int_id'].tolist())
    assert not is_an_integer_list(reverse_df['str_id'].tolist())
    assert is_an_integer_list(reverse_df['int_random'].tolist())

def int_formatter_df():
    row_cnt = 1000
    header = ['int_id', 'str_id', 'int_random', 'float_random']
    int_id = list(range(row_cnt))
    str_id = list(('id_' + str(i) for i in range(row_cnt)))
    int_random = np.random.randint(100, size=row_cnt)
    float_random = np.random.randn(row_cnt)
    X = [[int_id[i], str_id[i], int_random[i], float_random[i]] for i in range(row_cnt)]
    df = pd.DataFrame(X, columns=header)
    return df

def test_datetime_formatter_test_df(datetime_test_df: pd.DataFrame):
    """
    Test function for the DatetimeFormatter class.

    This function tests the functionality of the DatetimeFormatter class by creating a test DataFrame,
    setting the datetime format for the columns, fitting the transformer, converting the DataFrame,
    reversing the conversion, and checking if the reversed DataFrame is equal to the original one.

    Args:
        datetime_test_df (pd.DataFrame): The test DataFrame to be used for testing.

    Returns:
        None

    Raises:
        AssertionError: If any of the assertions fail.
    """
    assert datetime_test_df.shape == (1000, 7)
    metadata_df = Metadata.from_dataframe(datetime_test_df)
    assert metadata_df.datetime_columns == {'simple_datetime_2', 'date_with_time', 'simple_datetime'}
    datetime_format = {}
    datetime_format['simple_datetime'] = '%Y-%m-%d'
    datetime_format['simple_datetime_2'] = '%d %b %Y'
    datetime_format['date_with_time'] = '%Y-%m-%d %H:%M:%S'
    metadata_df.datetime_format = datetime_format
    transformer = DatetimeFormatter()
    assert not transformer.fitted
    transformer.fit(metadata=metadata_df)
    assert transformer.fitted
    assert transformer.dead_columns == []
    assert set(transformer.datetime_columns) == {'simple_datetime_2', 'date_with_time', 'simple_datetime'}
    converted_df = transformer.convert(datetime_test_df)
    assert is_an_integer_list(converted_df['date_with_time'].to_list())
    assert is_an_integer_list(converted_df['simple_datetime_2'].to_list())
    assert is_an_integer_list(converted_df['simple_datetime'].to_list())
    reverse_converte_df = transformer.reverse_convert(converted_df)
    assert is_a_string_list(reverse_converte_df['simple_datetime'].to_list())
    assert is_a_string_list(reverse_converte_df['date_with_time'].to_list())
    assert is_a_string_list(reverse_converte_df['simple_datetime_2'].to_list())
    assert reverse_converte_df.eq(datetime_test_df).all().all()

@pytest.fixture
def dummy_data(demo_relational_table_path):
    table_path_a, table_path_b, _ = demo_relational_table_path
    df_a = pd.read_csv(table_path_a)
    df_b = pd.read_csv(table_path_b)
    yield [(df_a, 'parent', Metadata.from_dataframe(df_a)), (df_b, 'child', Metadata.from_dataframe(df_b))]


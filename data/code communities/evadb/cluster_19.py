# Cluster 19

class DummyObjectDetectorDecorators(AbstractClassifierFunction):

    @decorators.setup(cacheable=True, function_type='object_detection', batchable=True)
    def setup(self, *args, **kwargs):
        pass

    @property
    def name(self) -> str:
        return 'DummyObjectDetectorDecorators'

    @property
    def labels(self):
        return ['__background__', 'person', 'bicycle']

    @decorators.forward(input_signatures=[PandasDataframe(columns=['Frame_Array'], column_types=[NdArrayType.UINT8], column_shapes=[(3, 256, 256)])], output_signatures=[NumpyArray(name='label', type=NdArrayType.STR)])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = pd.DataFrame()
        ret['label'] = df.apply(self.classify_one, axis=1)
        return ret

    def classify_one(self, frames: np.ndarray):
        i = int(frames[0][0][0][0]) - 1
        label = self.labels[i % 2 + 1]
        return np.array([label])

class DummyNoInputFunction(AbstractFunction):

    @decorators.setup(cacheable=False, function_type='test', batchable=False)
    def setup(self, *args, **kwargs):
        pass

    @property
    def name(self) -> str:
        return 'DummyNoInputFunction'

    @decorators.forward(input_signatures=[], output_signatures=[PandasDataframe(columns=['label'], column_types=[NdArrayType.STR], column_shapes=[(None,)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = pd.DataFrame([{'label': 'DummyNoInputFunction'}])
        return ret

class DummyLLM(AbstractFunction):

    @property
    def name(self) -> str:
        return 'DummyLLM'

    @decorators.setup(cacheable=True, function_type='chat-completion', batchable=True)
    def setup(self, *args, **kwargs):
        pass

    @decorators.forward(input_signatures=[PandasDataframe(columns=['query', 'content', 'prompt'], column_types=[NdArrayType.STR, NdArrayType.STR, NdArrayType.STR], column_shapes=[(1,), (1,), (None,)])], output_signatures=[PandasDataframe(columns=['response'], column_types=[NdArrayType.STR], column_shapes=[(1,)])])
    def forward(self, text_df):
        queries = text_df[text_df.columns[0]]
        content = text_df[text_df.columns[0]]
        if len(text_df.columns) > 1:
            queries = text_df.iloc[:, 0]
            content = text_df.iloc[:, 1]
        prompt = None
        if len(text_df.columns) > 2:
            prompt = text_df.iloc[0, 2]
        results = []
        for query, content in zip(queries, content):
            results.append(('' if prompt is None else prompt) + query + ' ' + content)
        df = pd.DataFrame({'response': results})
        time.sleep(1)
        return df

class DecoratorTests(unittest.TestCase):

    def test_setup_flags_are_updated(self):

        @setup(cacheable=True, function_type='classification', batchable=True)
        def setup_func():
            pass
        setup_func()
        self.assertTrue(setup_func.tags['cacheable'])
        self.assertTrue(setup_func.tags['batchable'])
        self.assertEqual(setup_func.tags['function_type'], 'classification')

    def test_setup_flags_are_updated_with_default_values(self):

        @setup()
        def setup_func():
            pass
        setup_func()
        self.assertFalse(setup_func.tags['cacheable'])
        self.assertTrue(setup_func.tags['batchable'])
        self.assertEqual(setup_func.tags['function_type'], 'Abstract')

    def test_forward_flags_are_updated(self):
        input_type = PandasDataframe(columns=['Frame_Array'], column_types=[NdArrayType.UINT8], column_shapes=[(3, 256, 256)])
        output_type = NumpyArray(name='label', type=NdArrayType.STR)

        @forward(input_signatures=[input_type], output_signatures=[output_type])
        def forward_func():
            pass
        forward_func()
        self.assertEqual(forward_func.tags['input'], [input_type])
        self.assertEqual(forward_func.tags['output'], [output_type])

def forward(input_signatures: List[IOArgument], output_signatures: List[IOArgument]):
    """decorator for the forward function. It will be used to set the input and output.

    Args:
        input_signature (List[IOArgument]): List of input arguments for the function
        output_signature ( List[IOArgument])): List of output arguments for the function
    """

    def inner_fn(arg_fn):

        def wrapper(*args):
            return arg_fn(*args)
        tags = {}
        tags['input'] = input_signatures
        tags['output'] = output_signatures
        wrapper.tags = tags
        return wrapper
    return inner_fn

class FunctionIODescriptorsTests(unittest.TestCase):

    def test_catalog_entry_for_numpy_entry(self):
        numpy_array = NumpyArray(name='input', is_nullable=False, type=NdArrayType.UINT8, dimensions=(2, 2))
        catalog_entries = numpy_array.generate_catalog_entries()
        self.assertEqual(len(catalog_entries), 1)
        catalog_entry = catalog_entries[0]
        self.assertEqual(catalog_entry.name, 'input')
        self.assertEqual(catalog_entry.type, ColumnType.NDARRAY)
        self.assertEqual(catalog_entry.is_nullable, False)
        self.assertEqual(catalog_entry.array_type, NdArrayType.UINT8)
        self.assertEqual(catalog_entry.array_dimensions, (2, 2))
        self.assertEqual(catalog_entry.is_input, False)

    def test_catalog_entry_for_pytorch_entry(self):
        pytorch_tensor = PyTorchTensor(name='input', is_nullable=False, type=NdArrayType.UINT8, dimensions=(2, 2))
        catalog_entries = pytorch_tensor.generate_catalog_entries()
        self.assertEqual(len(catalog_entries), 1)
        catalog_entry = catalog_entries[0]
        self.assertEqual(catalog_entry.name, 'input')
        self.assertEqual(catalog_entry.type, ColumnType.NDARRAY)
        self.assertEqual(catalog_entry.is_nullable, False)
        self.assertEqual(catalog_entry.array_type, NdArrayType.UINT8)
        self.assertEqual(catalog_entry.array_dimensions, (2, 2))
        self.assertEqual(catalog_entry.is_input, False)

    def test_catalog_entry_for_pandas_entry_with_single_column_simple(self):
        pandas_dataframe = PandasDataframe(columns=['Frame_Array'])
        catalog_entries = pandas_dataframe.generate_catalog_entries()
        self.assertEqual(len(catalog_entries), 1)
        catalog_entry = catalog_entries[0]
        self.assertEqual(catalog_entry.name, 'Frame_Array')
        self.assertEqual(catalog_entry.type, ColumnType.NDARRAY)
        self.assertEqual(catalog_entry.is_nullable, False)
        self.assertEqual(catalog_entry.array_type, NdArrayType.ANYTYPE)
        self.assertEqual(catalog_entry.array_dimensions, Dimension.ANYDIM)

    def test_catalog_entry_for_pandas_entry_with_single_column(self):
        pandas_dataframe = PandasDataframe(columns=['Frame_Array'], column_types=[NdArrayType.UINT8], column_shapes=[(3, 256, 256)])
        catalog_entries = pandas_dataframe.generate_catalog_entries()
        self.assertEqual(len(catalog_entries), 1)
        catalog_entry = catalog_entries[0]
        self.assertEqual(catalog_entry.name, 'Frame_Array')
        self.assertEqual(catalog_entry.type, ColumnType.NDARRAY)
        self.assertEqual(catalog_entry.is_nullable, False)
        self.assertEqual(catalog_entry.array_type, NdArrayType.UINT8)
        self.assertEqual(catalog_entry.array_dimensions, (3, 256, 256))
        self.assertEqual(catalog_entry.is_input, False)

    def test_catalog_entry_for_pandas_entry_with_multiple_columns_simple(self):
        pandas_dataframe = PandasDataframe(columns=['Frame_Array', 'Frame_Array_2'])
        catalog_entries = pandas_dataframe.generate_catalog_entries()
        self.assertEqual(len(catalog_entries), 2)
        catalog_entry = catalog_entries[0]
        self.assertEqual(catalog_entry.name, 'Frame_Array')
        self.assertEqual(catalog_entry.type, ColumnType.NDARRAY)
        self.assertEqual(catalog_entry.is_nullable, False)
        self.assertEqual(catalog_entry.array_type, NdArrayType.ANYTYPE)
        self.assertEqual(catalog_entry.array_dimensions, Dimension.ANYDIM)
        catalog_entry = catalog_entries[1]
        self.assertEqual(catalog_entry.name, 'Frame_Array_2')
        self.assertEqual(catalog_entry.type, ColumnType.NDARRAY)
        self.assertEqual(catalog_entry.is_nullable, False)
        self.assertEqual(catalog_entry.array_type, NdArrayType.ANYTYPE)
        self.assertEqual(catalog_entry.array_dimensions, Dimension.ANYDIM)

    def test_catalog_entry_for_pandas_entry_with_multiple_columns(self):
        pandas_dataframe = PandasDataframe(columns=['Frame_Array', 'Frame_Array_2'], column_types=[NdArrayType.UINT8, NdArrayType.FLOAT32], column_shapes=[(3, 256, 256), (3, 256, 256)])
        catalog_entries = pandas_dataframe.generate_catalog_entries()
        self.assertEqual(len(catalog_entries), 2)
        catalog_entry = catalog_entries[0]
        self.assertEqual(catalog_entry.name, 'Frame_Array')
        self.assertEqual(catalog_entry.type, ColumnType.NDARRAY)
        self.assertEqual(catalog_entry.is_nullable, False)
        self.assertEqual(catalog_entry.array_type, NdArrayType.UINT8)
        self.assertEqual(catalog_entry.array_dimensions, (3, 256, 256))
        self.assertEqual(catalog_entry.is_input, False)
        catalog_entry = catalog_entries[1]
        self.assertEqual(catalog_entry.name, 'Frame_Array_2')
        self.assertEqual(catalog_entry.type, ColumnType.NDARRAY)
        self.assertEqual(catalog_entry.is_nullable, False)
        self.assertEqual(catalog_entry.array_type, NdArrayType.FLOAT32)
        self.assertEqual(catalog_entry.array_dimensions, (3, 256, 256))
        self.assertEqual(catalog_entry.is_input, False)

    def test_raises_error_on_incorrect_pandas_definition(self):
        pandas_dataframe = PandasDataframe(columns=['Frame_Array', 'Frame_Array_2'], column_types=[NdArrayType.UINT8], column_shapes=[(3, 256, 256), (3, 256, 256)])
        with self.assertRaises(FunctionIODefinitionError):
            pandas_dataframe.generate_catalog_entries()

class ChatGPT(AbstractFunction):
    """
    Arguments:
        model (str) : ID of the OpenAI model to use. Refer to '_VALID_CHAT_COMPLETION_MODEL' for a list of supported models.
        temperature (float) : Sampling temperature to use in the model. Higher value results in a more random output.

    Input Signatures:
        query (str)   : The task / question that the user wants the model to accomplish / respond.
        content (str) : Any relevant context that the model can use to complete its tasks and generate the response.
        prompt (str)  : An optional prompt that can be passed to the model. It can contain instructions to the model,
                        or a set of examples to help the model generate a better response.
                        If not provided, the system prompt defaults to that of an helpful assistant that accomplishes user tasks.

    Output Signatures:
        response (str) : Contains the response generated by the model based on user input. Any errors encountered
                         will also be passed in the response.

    Example Usage:
        Assume we have the transcripts for a few videos stored in a table 'video_transcripts' in a column named 'text'.
        If the user wants to retrieve the summary of each video, the ChatGPT function can be used as:

            query = "Generate the summary of the video"
            cursor.table("video_transcripts").select(f"ChatGPT({query}, text)")

        In the above function invocation, the 'query' passed would be the user task to generate video summaries, and the
        'content' passed would be the video transcripts that need to be used in order to generate the summary. Since
        no prompt is passed, the default system prompt will be used.

        Now assume the user wants to create the video summary in 50 words and in French. Instead of passing these instructions
        along with each query, a prompt can be set as such:

            prompt = "Generate your responses in 50 words or less. Also, generate the response in French."
            cursor.table("video_transcripts").select(f"ChatGPT({query}, text, {prompt})")

        In the above invocation, an additional argument is passed as prompt. While the query and content arguments remain
        the same, the 'prompt' argument will be set as a system message in model params.

        Both of the above cases would generate a summary for each row / video transcript of the table in the response.
    """

    @property
    def name(self) -> str:
        return 'ChatGPT'

    @setup(cacheable=True, function_type='chat-completion', batchable=True)
    def setup(self, model='gpt-3.5-turbo', temperature: float=0, openai_api_key='') -> None:
        assert model in _VALID_CHAT_COMPLETION_MODEL, f'Unsupported ChatGPT {model}'
        self.model = model
        self.temperature = temperature
        self.openai_api_key = openai_api_key

    @forward(input_signatures=[PandasDataframe(columns=['query', 'content', 'prompt'], column_types=[NdArrayType.STR, NdArrayType.STR, NdArrayType.STR], column_shapes=[(1,), (1,), (None,)])], output_signatures=[PandasDataframe(columns=['response'], column_types=[NdArrayType.STR], column_shapes=[(1,)])])
    def forward(self, text_df):
        try_to_import_openai()
        from openai import OpenAI
        api_key = self.openai_api_key
        if len(self.openai_api_key) == 0:
            api_key = os.environ.get('OPENAI_API_KEY', '')
        assert len(api_key) != 0, "Please set your OpenAI API key using SET OPENAI_API_KEY = 'sk-' or environment variable (OPENAI_API_KEY)"
        client = OpenAI(api_key=api_key)

        @retry(tries=6, delay=20)
        def completion_with_backoff(**kwargs):
            return client.chat.completions.create(**kwargs)
        queries = text_df[text_df.columns[0]]
        content = text_df[text_df.columns[0]]
        if len(text_df.columns) > 1:
            queries = text_df.iloc[:, 0]
            content = text_df.iloc[:, 1]
        prompt = None
        if len(text_df.columns) > 2:
            prompt = text_df.iloc[0, 2]
        results = []
        for query, content in zip(queries, content):
            params = {'model': self.model, 'temperature': self.temperature, 'messages': []}
            def_sys_prompt_message = {'role': 'system', 'content': prompt if prompt is not None else 'You are a helpful assistant that accomplishes user tasks.'}
            params['messages'].append(def_sys_prompt_message)
            params['messages'].extend([{'role': 'user', 'content': f'Here is some context : {content}'}, {'role': 'user', 'content': f'Complete the following task: {query}'}])
            response = completion_with_backoff(**params)
            answer = response.choices[0].message.content
            results.append(answer)
        df = pd.DataFrame({'response': results})
        return df

def try_to_import_openai():
    try:
        import openai
    except ImportError:
        raise ValueError('Could not import openai python package.\n                Please install them with `pip install openai`.')

class SiftFeatureExtractor(AbstractFunction, GPUCompatible):

    @setup(cacheable=False, function_type='FeatureExtraction', batchable=False)
    def setup(self):
        try_to_import_kornia()
        import kornia
        self.model = kornia.feature.SIFTDescriptor(100)

    def to_device(self, device: str) -> GPUCompatible:
        self.model = self.model.to(device)
        return self

    @property
    def name(self) -> str:
        return 'SiftFeatureExtractor'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.UINT8], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['features'], column_types=[NdArrayType.FLOAT32], column_shapes=[(1, 128)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            rgb_img = row[0]
            try_to_import_cv2()
            import cv2
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            resized_gray_img = cv2.resize(gray_img, (100, 100), interpolation=cv2.INTER_AREA)
            resized_gray_img = np.moveaxis(resized_gray_img, -1, 0)
            batch_resized_gray_img = np.expand_dims(resized_gray_img, axis=0)
            batch_resized_gray_img = np.expand_dims(batch_resized_gray_img, axis=0)
            batch_resized_gray_img = batch_resized_gray_img.astype(np.float32)
            try_to_import_torch()
            import torch
            with torch.no_grad():
                torch_feat = self.model(torch.from_numpy(batch_resized_gray_img))
                feat = torch_feat.numpy()
            feat = feat.reshape(1, -1)
            return feat
        ret = pd.DataFrame()
        ret['features'] = df.apply(_forward, axis=1)
        return ret

class FuzzDistance(AbstractFunction):

    @setup(cacheable=False, function_type='FeatureExtraction', batchable=False)
    def setup(self):
        pass

    @property
    def name(self) -> str:
        return 'FuzzDistance'

    @forward(input_signatures=[PandasDataframe(columns=['data1', 'data2'], column_types=[NdArrayType.STR, NdArrayType.STR], column_shapes=[1, 1])], output_signatures=[PandasDataframe(columns=['distance'], column_types=[NdArrayType.FLOAT32], column_shapes=[1])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            data1 = row.iloc[0]
            data2 = row.iloc[1]
            distance = fuzz.ratio(data1, data2)
            return distance
        ret = pd.DataFrame()
        ret['distance'] = df.apply(_forward, axis=1)
        return ret

class FastRCNNObjectDetector(PytorchAbstractClassifierFunction):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score

    """

    @property
    def name(self) -> str:
        return 'fastrcnn'

    @setup(cacheable=True, function_type='object_detection', batchable=True)
    def setup(self, threshold=0.85):
        try_to_import_torch()
        try_to_import_torchvision()
        import torchvision
        self.threshold = threshold
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1', progress=False)
        self.model.eval()

    @property
    def labels(self) -> List[str]:
        return ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    @forward(input_signatures=[PyTorchTensor(name='input_col', is_nullable=False, type=NdArrayType.FLOAT32, dimensions=(1, 3, 540, 960))], output_signatures=[PandasDataframe(columns=['labels', 'bboxes', 'scores'], column_types=[NdArrayType.STR, NdArrayType.FLOAT32, NdArrayType.FLOAT32], column_shapes=[(None,), (None,), (None,)])])
    def forward(self, frames) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed

        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])

        """
        predictions = self.model(frames)
        outcome = []
        for prediction in predictions:
            pred_class = [str(self.labels[i]) for i in list(self.as_numpy(prediction['labels']))]
            pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(self.as_numpy(prediction['boxes']))]
            pred_score = list(self.as_numpy(prediction['scores']))
            valid_pred = [pred_score.index(x) for x in pred_score if x > self.threshold]
            if valid_pred:
                pred_t = valid_pred[-1]
            else:
                pred_t = -1
            pred_boxes = np.array(pred_boxes[:pred_t + 1])
            pred_class = np.array(pred_class[:pred_t + 1])
            pred_score = np.array(pred_score[:pred_t + 1])
            outcome.append({'labels': pred_class, 'scores': pred_score, 'bboxes': pred_boxes})
        return pd.DataFrame(outcome, columns=['labels', 'scores', 'bboxes'])

class StableDiffusion(AbstractFunction):

    @property
    def name(self) -> str:
        return 'StableDiffusion'

    def setup(self, replicate_api_token='') -> None:
        self.replicate_api_token = replicate_api_token

    @forward(input_signatures=[PandasDataframe(columns=['prompt'], column_types=[NdArrayType.STR], column_shapes=[(None,)])], output_signatures=[PandasDataframe(columns=['response'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, text_df):
        try_to_import_replicate()
        import replicate
        replicate_api_key = self.replicate_api_token
        if replicate_api_key is None:
            replicate_api_key = os.environ.get('REPLICATE_API_TOKEN', '')
        assert len(replicate_api_key) != 0, "Please set your Replicate API key using SET REPLICATE_API_TOKEN = '' or set the environment variable (REPLICATE_API_TOKEN)"
        os.environ['REPLICATE_API_TOKEN'] = replicate_api_key
        model_id = replicate.models.get('stability-ai/stable-diffusion').versions.list()[0].id

        def generate_image(text_df: PandasDataframe):
            results = []
            queries = text_df[text_df.columns[0]]
            for query in queries:
                output = replicate.run('stability-ai/stable-diffusion:' + model_id, input={'prompt': query})
                response = requests.get(output[0])
                image = Image.open(BytesIO(response.content))
                frame = np.array(image)
                results.append(frame)
            return results
        df = pd.DataFrame({'response': generate_image(text_df=text_df)})
        return df

class SaliencyFeatureExtractor(AbstractFunction, GPUCompatible):

    @setup(cacheable=False, function_type='FeatureExtraction', batchable=False)
    def setup(self):
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()

    def to_device(self, device: str) -> GPUCompatible:
        self.model = self.model.to(device)
        return self

    @property
    def name(self) -> str:
        return 'SaliencyFeatureExtractor'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.UINT8], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['saliency'], column_types=[NdArrayType.FLOAT32], column_shapes=[(1, 224, 224)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            rgb_img = row[0]
            composed = Compose([Resize((224, 224)), ToTensor()])
            transformed_img = composed(Image.fromarray(rgb_img[:, :, ::-1])).unsqueeze(0)
            transformed_img.requires_grad_()
            outputs = self.model(transformed_img)
            score_max_index = outputs.argmax()
            score_max = outputs[0, score_max_index]
            score_max.backward()
            saliency, _ = torch.max(transformed_img.grad.data.abs(), dim=1)
            return saliency
        ret = pd.DataFrame()
        ret['saliency'] = df.apply(_forward, axis=1)
        return ret

class Yolo(AbstractFunction, GPUCompatible):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    @property
    def name(self) -> str:
        return 'yolo'

    @setup(cacheable=True, function_type='object_detection', batchable=True)
    def setup(self, model: str, threshold=0.3):
        try_to_import_ultralytics()
        from ultralytics import YOLO
        self.threshold = threshold
        self.model = YOLO(model)
        self.device = 'cpu'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['labels', 'bboxes', 'scores'], column_types=[NdArrayType.STR, NdArrayType.FLOAT32, NdArrayType.FLOAT32], column_shapes=[(None,), (None,), (None,)])])
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed
        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])
        """
        outcome = []
        frames = np.ravel(frames.to_numpy())
        list_of_numpy_images = [its for its in frames]
        predictions = self.model.predict(list_of_numpy_images, device=self.device, conf=self.threshold, verbose=False)
        for pred in predictions:
            single_result = pred.boxes
            pred_class = [self.model.names[i] for i in single_result.cls.tolist()]
            pred_score = single_result.conf.tolist()
            pred_score = [round(conf, 2) for conf in single_result.conf.tolist()]
            pred_boxes = single_result.xyxy.tolist()
            sorted_list = list(map(lambda i: i < self.threshold, pred_score))
            t = sorted_list.index(True) if True in sorted_list else len(sorted_list)
            outcome.append({'labels': pred_class[:t], 'bboxes': pred_boxes[:t], 'scores': pred_score[:t]})
        return pd.DataFrame(outcome, columns=['labels', 'bboxes', 'scores'])

    def to_device(self, device: str):
        self.device = device
        return self

class TextFilterKeyword(AbstractFunction):

    @setup(cacheable=False, function_type='TextProcessing', batchable=False)
    def setup(self):
        pass

    @property
    def name(self) -> str:
        return 'TextFilterKeyword'

    @forward(input_signatures=[PandasDataframe(columns=['data', 'keyword'], column_types=[NdArrayType.STR, NdArrayType.STR], column_shapes=[1, 1])], output_signatures=[PandasDataframe(columns=['filtered'], column_types=[NdArrayType.STR], column_shapes=[1])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            import re
            data = row.iloc[0]
            keywords = row.iloc[1]
            flag = False
            for i in keywords:
                pattern = f'^(.*?({i})[^$]*)$'
                match_check = re.search(pattern, data, re.IGNORECASE)
                if match_check:
                    flag = True
            if flag is False:
                return data
            flag = False
        ret = pd.DataFrame()
        ret['filtered'] = df.apply(_forward, axis=1)
        return ret

class DallEFunction(AbstractFunction):

    @property
    def name(self) -> str:
        return 'DallE'

    def setup(self, openai_api_key='') -> None:
        self.openai_api_key = openai_api_key

    @forward(input_signatures=[PandasDataframe(columns=['prompt'], column_types=[NdArrayType.STR], column_shapes=[(None,)])], output_signatures=[PandasDataframe(columns=['response'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, text_df):
        try_to_import_openai()
        from openai import OpenAI
        api_key = self.openai_api_key
        if len(self.openai_api_key) == 0:
            api_key = os.environ.get('OPENAI_API_KEY', '')
        assert len(api_key) != 0, "Please set your OpenAI API key using SET OPENAI_API_KEY = 'sk-' or environment variable (OPENAI_API_KEY)"
        client = OpenAI(api_key=api_key)

        def generate_image(text_df: PandasDataframe):
            results = []
            queries = text_df[text_df.columns[0]]
            for query in queries:
                response = client.images.generate(prompt=query, n=1, size='1024x1024')
                image_response = requests.get(response.data[0].url)
                image = Image.open(BytesIO(image_response.content))
                frame = np.array(image)
                results.append(frame)
            return results
        df = pd.DataFrame({'response': generate_image(text_df=text_df)})
        return df

class SentenceTransformerFeatureExtractor(AbstractFunction, GPUCompatible):

    @setup(cacheable=False, function_type='FeatureExtraction', batchable=False)
    def setup(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def to_device(self, device: str) -> GPUCompatible:
        self.model = self.model.to(device)
        return self

    @property
    def name(self) -> str:
        return 'SentenceTransformerFeatureExtractor'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.STR], column_shapes=[1])], output_signatures=[PandasDataframe(columns=['features'], column_types=[NdArrayType.FLOAT32], column_shapes=[(1, 384)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        def _forward(row: pd.Series) -> np.ndarray:
            data = row
            embedded_list = self.model.encode(data)
            return embedded_list
        ret = pd.DataFrame()
        ret['features'] = df.apply(_forward, axis=1)
        return ret

class GaussianBlur(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        pass

    @property
    def name(self):
        return 'GaussianBlur'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['blurred_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Gaussian Blur to the frame

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def gaussianBlur(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            try_to_import_cv2()
            import cv2
            frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        ret = pd.DataFrame()
        ret['blurred_frame_array'] = frame.apply(gaussianBlur, axis=1)
        return ret

class Annotate(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        pass

    @property
    def name(self):
        return 'Annotate'

    @forward(input_signatures=[PandasDataframe(columns=['data', 'labels', 'bboxes'], column_types=[NdArrayType.FLOAT32, NdArrayType.STR, NdArrayType.FLOAT32], column_shapes=[(None, None, 3), (None,), (None,)])], output_signatures=[PandasDataframe(columns=['annotated_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modify the frame to annotate the bbox on it.

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def annotate(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            bboxes = row[2]
            try_to_import_cv2()
            import cv2
            for bbox in bboxes:
                x1, y1, x2, y2 = np.asarray(bbox, dtype='int')
                x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            return frame
        ret = pd.DataFrame()
        ret['annotated_frame_array'] = df.apply(annotate, axis=1)
        return ret

class VerticalFlip(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        try_to_import_cv2()

    @property
    def name(self):
        return 'VerticalFlip'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['vertically_flipped_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply vertical flip to the frame

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def verticalFlip(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            try_to_import_cv2()
            import cv2
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        ret = pd.DataFrame()
        ret['vertically_flipped_frame_array'] = frame.apply(verticalFlip, axis=1)
        return ret

class HorizontalFlip(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        try_to_import_cv2()

    @property
    def name(self):
        return 'HorizontalFlip'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['horizontally_flipped_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply horizontal flip to the frame

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def horizontalFlip(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            try_to_import_cv2()
            import cv2
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        ret = pd.DataFrame()
        ret['horizontally_flipped_frame_array'] = frame.apply(horizontalFlip, axis=1)
        return ret

class ToGrayscale(AbstractFunction):

    @setup(cacheable=False, function_type='cv2-transformation', batchable=True)
    def setup(self):
        try_to_import_cv2()

    @property
    def name(self):
        return 'ToGrayscale'

    @forward(input_signatures=[PandasDataframe(columns=['data'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])], output_signatures=[PandasDataframe(columns=['grayscale_frame_array'], column_types=[NdArrayType.FLOAT32], column_shapes=[(None, None, 3)])])
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the frame from BGR to grayscale

         Returns:
             ret (pd.DataFrame): The modified frame.
        """

        def toGrayscale(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            return frame
        ret = pd.DataFrame()
        ret['grayscale_frame_array'] = frame.apply(toGrayscale, axis=1)
        return ret

class EvaDBTrackerAbstractFunction(AbstractFunction):
    """
    An abstract class for all EvaDB object trackers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @setup(cacheable=False, function_type='object_tracker', batchable=False)
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

    @forward(input_signatures=[PandasDataframe(columns=['frame_id', 'frame', 'bboxes', 'scores', 'labels'], column_types=[NdArrayType.INT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32, NdArrayType.STR], column_shapes=[(1,), (None, None, 3), (None, 4), (None,), (None,)])], output_signatures=[PandasDataframe(columns=['track_ids', 'track_labels', 'track_bboxes', 'track_scores'], column_types=[NdArrayType.INT32, NdArrayType.INT32, NdArrayType.FLOAT32, NdArrayType.FLOAT32], column_shapes=[(None,), (None,), (None, 4), (None,)])])
    def forward(self, frame_id: numpy.ndarray, frame: numpy.ndarray, labels: numpy.ndarray, bboxes: numpy.ndarray, scores: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Args:
            frame_id (numpy.ndarray): the frame id of current frame
            frame (numpy.ndarray): the input frame with shape (C, H, W)
            labels (numpy.ndarray): Corresponding labels for each box
            bboxes (numpy.ndarray): Array of shape `(n, 4)` or of shape `(4,)` where
            each row contains `(xmin, ymin, width, height)`.
            scores (numpy.ndarray): Corresponding scores for each box
        Returns:
            track_ids (numpy.ndarray): Corresponding track id for each box
            track_labels (numpy.ndarray): Corresponding labels for each box
            track_bboxes (numpy.ndarray):  Array of shape `(n, 4)` of tracked objects
            track_scores (numpy.ndarray): Corresponding scores for each box
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        assert isinstance(args[0], pd.DataFrame), f'Expecting pd.DataFrame, got {type(args[0])}'
        results = []
        for _, row in args[0].iterrows():
            tuple = (numpy.array(row[0]), numpy.array(row[1]), numpy.stack(row[2]), numpy.stack(row[3]), numpy.stack(row[4]))
            results.append(self.forward(*tuple))
        return pd.DataFrame(results, columns=['track_ids', 'track_labels', 'track_bboxes', 'track_scores'])


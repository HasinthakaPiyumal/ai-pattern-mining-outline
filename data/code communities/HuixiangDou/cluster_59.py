# Cluster 59

class FeatureStore:
    """Tokenize and extract features from the project's documents, for use in
    the reject pipeline and response pipeline."""

    def __init__(self, embedder: Embedder, config_path: str='config.ini', language: str='zh', chunk_size=900, analyze_reject=False, rejecter_naive_splitter=False, override=False) -> None:
        """Init with model device type and config."""
        self.config_path = config_path
        self.reject_throttle = -1
        self.language = language
        self.override = override
        with open(config_path, encoding='utf8') as f:
            config = pytoml.load(f)['feature_store']
            self.reject_throttle = config['reject_throttle']
        logger.debug('loading text2vec model..')
        self.embedder = embedder
        self.retriever = None
        self.chunk_size = chunk_size
        self.analyze_reject = analyze_reject
        if rejecter_naive_splitter:
            raise ValueError('The `rejecter_naive_splitter` option deprecated, please `git checkout v20240722`')
        if analyze_reject:
            raise ValueError('The `analyze_reject` option deprecated, please `git checkout v20240722`')
        logger.info('init dense retrieval database with chunk_size {}'.format(chunk_size))
        if language == 'zh':
            self.text_splitter = ChineseRecursiveTextSplitter(keep_separator=True, is_separator_regex=True, chunk_size=chunk_size, chunk_overlap=32)
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=32)

    def parse_markdown(self, file: FileName, metadata: Dict):
        length = 0
        text = file.basename + '\n'
        with open(file.copypath, encoding='utf8') as f:
            text += f.read()
        if len(text) <= 1:
            return ([], length)
        chunks = nested_split_markdown(file.origin, text=text, chunksize=self.chunk_size, metadata=metadata)
        for c in chunks:
            length += len(c.content_or_path)
        return (chunks, length)

    def build_inverted_index(self, chunks: List[Chunk], ner_file: str, work_dir: str):
        """Build inverted index based on named entity for knowledge base."""
        if ner_file is None:
            return
        index_dir = os.path.join(work_dir, 'db_reverted_index')
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        entities = []
        with open(ner_file) as f:
            entities = json.load(f)
        time0 = time.time()
        map_entity2chunks = dict()
        indexer = NamedEntity2Chunk(file_dir=index_dir)
        indexer.clean()
        indexer.set_entity(entities=entities)
        for chunk_id, chunk in enumerate(chunks):
            if chunk.modal != 'text':
                continue
            entity_ids = indexer.parse(text=chunk.content_or_path)
            for entity_id in entity_ids:
                if entity_id not in map_entity2chunks:
                    map_entity2chunks[entity_id] = [chunk_id]
                else:
                    map_entity2chunks[entity_id].append(chunk_id)
        for entity_id, chunk_indexes in map_entity2chunks.items():
            indexer.insert_relation(eid=entity_id, chunk_ids=chunk_indexes)
        del indexer
        time1 = time.time()
        logger.info('Timecost for build_inverted_index {}s'.format(time1 - time0))

    def build_sparse(self, files: List[FileName], work_dir: str):
        """Use BM25 for building code feature"""
        fileopr = FileOperation()
        chunks = []
        for file in files:
            content, error = fileopr.read(file.origin)
            if error is not None:
                continue
            file_chunks = split_python_code(filepath=file.origin, text=content, metadata={'source': file.origin, 'read': file.copypath})
            chunks += file_chunks
        sparse_dir = os.path.join(work_dir, 'db_sparse')
        bm25 = BM25Okapi()
        bm25.save(chunks, sparse_dir)

    def process_qa_pairs(self, qa_pair_file: str) -> List[Chunk]:
        """Process QA pairs from CSV or JSON file.
        
        Args:
            qa_pair_file: Path to the CSV or JSON file containing QA pairs.
            
        Returns:
            List of Chunk objects where key is the content and value is stored in metadata.
        """
        chunks = []
        file_ext = os.path.splitext(qa_pair_file)[1].lower()
        try:
            if file_ext == '.csv':
                with open(qa_pair_file, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        if len(row) >= 2:
                            key, value = (row[0], row[1])
                            chunk = Chunk(modal='qa', content_or_path=key, metadata={'read': qa_pair_file, 'source': qa_pair_file, 'qa': f'{key}: {value}'})
                            chunks.append(chunk)
            elif file_ext == '.json':
                with open(qa_pair_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                    if isinstance(qa_data, dict):
                        for key, value in qa_data.items():
                            chunk = Chunk(modal='qa', content_or_path=key, metadata={'read': qa_pair_file, 'source': qa_pair_file, 'qa': f'{key}: {value}'})
                            chunks.append(chunk)
                    elif isinstance(qa_data, list):
                        for item in qa_data:
                            if isinstance(item, dict) and 'key' in item and ('value' in item):
                                chunk = Chunk(modal='qa', content_or_path=key, metadata={'read': qa_pair_file, 'source': qa_pair_file, 'qa': f'{key}: {value}'})
                                chunks.append(chunk)
            logger.info(f'Processed {len(chunks)} QA pairs from {qa_pair_file}')
            return chunks
        except Exception as e:
            logger.error(f'Error processing QA pairs from {qa_pair_file}: {str(e)}')
            return []

    def build_dense(self, files: List[FileName], work_dir: str, markdown_as_txt: bool=False, qa_pair_file: str=None):
        """Extract the features required for the response pipeline based on the
        document."""
        feature_dir = os.path.join(work_dir, 'db_dense')
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        file_opr = FileOperation()
        chunks = []
        if qa_pair_file is not None:
            qa_chunks = self.process_qa_pairs(qa_pair_file)
            chunks.extend(qa_chunks)
            logger.info(f'Added {len(qa_chunks)} chunks from QA pairs')
        for i, file in tqdm(enumerate(files), 'split'):
            if not file.state:
                continue
            metadata = {'source': file.origin, 'read': file.copypath}
            if not markdown_as_txt and file._type == 'md':
                md_chunks, md_length = self.parse_markdown(file=file, metadata=metadata)
                chunks += md_chunks
                file.reason = str(md_length)
            else:
                text, error = file_opr.read(file.copypath)
                if error is not None:
                    file.state = False
                    file.reason = str(error)
                    continue
                file.reason = str(len(text))
                text = file.prefix + text
                chunks += self.text_splitter.create_chunks(texts=[text], metadatas=[metadata])
        if not self.embedder.support_image:
            filtered_chunks = list(filter(lambda x: x.modal == 'text' or x.modal == 'qa', chunks))
        else:
            filtered_chunks = chunks
        if len(chunks) < 1:
            return chunks
        self.analyze(filtered_chunks)
        Faiss.save_local(folder_path=feature_dir, chunks=filtered_chunks, embedder=self.embedder)
        return chunks

    def analyze(self, chunks: List[Chunk]):
        """Output documents length mean, median and histogram."""
        MAX_COUNT = 10000
        if len(chunks) > MAX_COUNT:
            chunks = random.sample(chunks, MAX_COUNT)
        text_lens = []
        token_lens = []
        text_chunk_count = 0
        image_chunk_count = 0
        if self.embedder is None:
            logger.info('self.embedder is None, skip `anaylze_output`')
            return
        for chunk in tqdm(chunks, 'analyze distribution'):
            if chunk.modal == 'image':
                image_chunk_count += 1
            elif chunk.modal == 'text':
                text_chunk_count += 1
            content = chunk.content_or_path
            text_lens.append(len(content))
            token_lens.append(self.embedder.token_length(content))
        logger.info('text_chunks {}, image_chunks {}'.format(text_chunk_count, image_chunk_count))
        logger.info('text histogram, {}'.format(histogram(text_lens)))
        logger.info('token histogram, {}'.format(histogram(token_lens)))

    def preprocess(self, files: List, work_dir: str):
        """Preprocesses files in a given directory. Copies each file to
        'preprocess' with new name formed by joining all subdirectories with
        '_'.

        Args:
            files (list): original file list.
            work_dir (str): Working directory where preprocessed files will be stored.  # noqa E501

        Returns:
            str: Path to the directory where preprocessed markdown files are saved.

        Raises:
            Exception: Raise an exception if no markdown files are found in the provided repository directory.  # noqa E501
        """
        preproc_dir = os.path.join(work_dir, 'preprocess')
        if not os.path.exists(preproc_dir):
            os.makedirs(preproc_dir)
        pool = Pool(processes=8)
        file_opr = FileOperation()
        for idx, file in tqdm(enumerate(files), 'preprocess'):
            if not os.path.exists(file.origin):
                file.state = False
                file.reason = 'skip not exist'
                continue
            if file._type == 'image':
                file.state = False
                file.reason = 'skip image'
            elif file._type in ['pdf', 'word', 'excel', 'ppt', 'html']:
                md5 = file_opr.md5(file.origin)
                file.copypath = os.path.join(preproc_dir, '{}.text'.format(md5))
                pool.apply_async(read_and_save, (file,))
            elif file._type in ['code']:
                md5 = file_opr.md5(file.origin)
                file.copypath = os.path.join(preproc_dir, '{}.code'.format(md5))
                read_and_save(file)
            elif file._type in ['md', 'text']:
                md5 = file_opr.md5(file.origin)
                file.copypath = os.path.join(preproc_dir, file.origin.replace('/', '_')[-84:])
                try:
                    shutil.copy(file.origin, file.copypath)
                    file.state = True
                    file.reason = 'preprocessed'
                except Exception as e:
                    file.state = False
                    file.reason = str(e)
            else:
                file.state = False
                file.reason = 'skip unknown format'
        pool.close()
        logger.debug('waiting for file preprocess finish..')
        pool.join()
        for file in files:
            if file._type in ['pdf', 'word', 'excel']:
                if os.path.exists(file.copypath):
                    file.state = True
                    file.reason = 'preprocessed'
                else:
                    file.state = False
                    file.reason = 'read error'

    def initialize(self, config: InitializeConfig):
        """Initializes response and reject feature store.

        Only needs to be called once. Also calculates the optimal threshold
        based on provided good and bad question examples, and saves it in the
        configuration file.
        
        Args:
            config: Configuration object containing initialization parameters
        """
        logger.info('initialize response and reject feature store, you only need call this once.')
        self.preprocess(files=config.files, work_dir=config.work_dir)
        documents = list(filter(lambda x: x._type != 'code', config.files))
        chunks = self.build_dense(files=documents, work_dir=config.work_dir, qa_pair_file=config.qa_pair_file)
        codes = list(filter(lambda x: x._type == 'code', config.files))
        self.build_sparse(files=codes, work_dir=config.work_dir)
        self.build_inverted_index(chunks=chunks, ner_file=config.ner_file, work_dir=config.work_dir)

def split_python_code(filepath: str, text: str, metadata: dict={}):
    """Split python code to class, function and annotation."""
    basename = os.path.basename(filepath)
    texts = []
    texts.append(basename)
    try:
        node = ast.parse(text)
        data = ast.get_docstring(node)
        if data:
            texts.append(data)
        for child_node in ast.walk(node):
            if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                data = ast.get_docstring(child_node)
                if data:
                    texts.append(f'{child_node.name} {data}')
    except Exception as e:
        logger.error('{} {}, continue'.format(filepath, str(e)))
    chunks = []
    for text in texts:
        chunks.append(Chunk(content_or_path=text, metadata=metadata))
    return chunks


# Cluster 94

class DocumentReader(AbstractReader):

    def __init__(self, *args, chunk_params, **kwargs):
        super().__init__(*args, **kwargs)
        self._LOADER_MAPPING = _lazy_import_loader()
        self._splitter_class = _lazy_import_text_splitter()
        self._chunk_size = chunk_params.get('chunk_size', DEFAULT_DOCUMENT_CHUNK_SIZE)
        self._chunk_overlap = chunk_params.get('chunk_overlap', DEFAULT_DOCUMENT_CHUNK_OVERLAP)

    def _read(self) -> Iterator[Dict]:
        ext = Path(self.file_url).suffix
        assert ext in self._LOADER_MAPPING, f'File Format {ext} not supported'
        loader_class, loader_args = self._LOADER_MAPPING[ext]
        loader = loader_class(self.file_url, **loader_args)
        langchain_text_splitter = self._splitter_class(chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        row_num = 0
        for data in loader.load():
            for chunk_id, row in enumerate(langchain_text_splitter.split_documents([data])):
                yield {'chunk_id': chunk_id, 'data': row.page_content, ROW_NUM_COLUMN: row_num}
                row_num += 1

def _lazy_import_loader():
    try_to_import_langchain()
    from langchain.document_loaders import CSVLoader, EverNoteLoader, PDFMinerLoader, TextLoader, UnstructuredEmailLoader, UnstructuredEPubLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
    LOADER_MAPPING = {'.doc': (UnstructuredWordDocumentLoader, {}), '.docx': (UnstructuredWordDocumentLoader, {}), '.enex': (EverNoteLoader, {}), '.eml': (UnstructuredEmailLoader, {}), '.epub': (UnstructuredEPubLoader, {}), '.html': (UnstructuredHTMLLoader, {}), '.csv': (CSVLoader, {}), '.md': (UnstructuredMarkdownLoader, {}), '.pdf': (PDFMinerLoader, {}), '.ppt': (UnstructuredPowerPointLoader, {}), '.pptx': (UnstructuredPowerPointLoader, {}), '.txt': (TextLoader, {'encoding': 'utf8'})}
    return LOADER_MAPPING

def _lazy_import_text_splitter():
    try_to_import_langchain()
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter

def try_to_import_langchain():
    try:
        import langchain
    except ImportError:
        raise ValueError('Could not import langchain package.\n                Please install it with `pip install langchain`.')


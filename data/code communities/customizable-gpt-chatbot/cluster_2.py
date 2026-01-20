# Cluster 2

def build_or_update_faiss_index(file_path, index_name):
    faiss_obj_path = os.path.join(MODELS_DIR, f'{index_name}.pickle')
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    loader = get_loader(file_path)
    pages = loader.load_and_split()
    if os.path.exists(faiss_obj_path):
        faiss_index = FAISS.load(faiss_obj_path)
        new_embeddings = FAISS.from_documents(pages, embeddings, index_name=index_name)
        faiss_index.add_vectors(new_embeddings)
    else:
        faiss_index = FAISS.from_documents(pages, embeddings, index_name=index_name)
    faiss_index.save(faiss_obj_path)
    return faiss_index

def get_loader(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type == 'application/pdf':
        return PyPDFLoader(file_path)
    elif mime_type == 'text/csv':
        return CSVLoader(file_path)
    elif mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f'Unsupported file type: {mime_type}')


# Cluster 4

def dynamic_upload_to(instance, filename):
    if instance.storage_type == 'FAISS':
        return upload_to_faiss(instance, filename)
    else:
        return upload_to_pinecone(instance, filename)

def upload_to_faiss(instance, filename):
    basename, ext = os.path.splitext(filename)
    new_filename = f'{basename}_{uuid.uuid4().hex}{ext}'
    return f'documents/faiss/{new_filename}'

def upload_to_pinecone(instance, filename):
    basename, ext = os.path.splitext(filename)
    new_filename = f'{basename}_{uuid.uuid4().hex}{ext}'
    return f'documents/pinecone/{new_filename}'


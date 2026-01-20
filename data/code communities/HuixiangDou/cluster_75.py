# Cluster 75

def langchain_splitter(text: str, chunk_size: int, metadata):
    """This is for debugging"""
    from langchain.text_splitter import MarkdownTextSplitter
    md_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=32)
    docs = md_splitter.create_documents([text])
    chunks = []
    for doc in docs:
        c = Chunk(content_or_path=doc.page_content, metadata=metadata)
        chunks.append(c)
    return chunks


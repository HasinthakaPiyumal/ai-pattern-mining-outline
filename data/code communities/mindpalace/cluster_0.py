# Cluster 0

def reset_state():
    st.session_state.content_generated = False
    st.session_state.extracted_text = None
    st.session_state.topic_data = None
    st.session_state.vectorstore = None
    st.session_state.messages = [SystemMessage('You are an assistant for question-answering tasks.')]
    st.session_state.mindmap_generated = False

def create_embeddings(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(data)
    embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


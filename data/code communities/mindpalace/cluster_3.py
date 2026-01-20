# Cluster 3

def process_pdf():
    if st.session_state.pdf:
        with st.spinner(text='fetching pdf contents', show_time=False):
            text = extract_from_mistral(st.session_state.pdf)
            if text.strip():
                st.session_state.extracted_text = text
                st.session_state.content_generated = True
                return True
            else:
                st.error('No text could be extracted from the PDF.')
    else:
        st.error('invalid pdf. please enter a valid pdf.')
    return False

def extract_from_mistral(pdf_file):
    pdf_bytes = pdf_file.read()
    encoded_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    document = {'type': 'document_url', 'document_url': f'data:application/pdf;base64,{encoded_pdf}'}
    try:
        ocr_response = client.ocr.process(model='mistral-ocr-latest', document=document, include_image_base64=True)
        time.sleep(1)
        pages = ocr_response.pages if hasattr(ocr_response, 'pages') else ocr_response if isinstance(ocr_response, list) else []
        result_text = '\n\n'.join((page.markdown for page in pages)) or 'No result found.'
    except Exception as e:
        result_text = f'Error extracting result: {e}'
        st.cache_data.clear()
    return result_text

def on_generate():
    st.session_state.content_generated = False
    st.session_state.extracted_text = None
    st.session_state.vectorstore = None
    st.session_state.messages = [SystemMessage('You are an assistant for question-answering tasks.')]
    st.session_state.mindmap_generated = False
    if os.path.exists('faiss_index'):
        import shutil
        shutil.rmtree('faiss_index')
    if st.session_state.input_option == 'github repository':
        process_github()
        with st.spinner('analysing repository'):
            st.session_state.topic_data = ss_repo_text(st.session_state.extracted_text)
    elif st.session_state.input_option == 'pdf document':
        process_pdf()
        with st.spinner('analysing pdf'):
            st.session_state.topic_data = ss_pdf_text(st.session_state.extracted_text)

@st.cache_data(show_spinner=False)
def ss_repo_text(text):
    model = genai.GenerativeModel(model_name='gemini-2.0-flash', generation_config=generation_config, system_instruction='\n        Extract **all relevant files** from the GitHub repository and generate a **detailed, structured summary**.\n\n        **Rules:**\n        - Identify **all important files** (code files, config files, documentation, assets, workflows).\n        - **Group files** into categories (e.g., Backend, Frontend, Configurations, Testing, Documentation).\n        - **Ignore unnecessary** files like `.gitignore`, `LICENSE`, `.DS_Store`, `CONTRIBUTING`\n        - For each file, provide:\n          1. **File Purpose** - Explain what the file does and its significance in the repository.\n          2. **Key Functions & Classes** - List all major functions, methods, and classes with descriptions.\n          3. **Dependencies** - Mention which other files or external libraries it interacts with.\n          4. **Execution Flow** - Describe how this file contributes to the overall project.\n          5. **Important Configurations or Constants** (if applicable).\n          6. **Data Handling** (if applicable) - How the file processes or transforms data.\n        - Ensure summaries are **very detailed (8-12 sentences per file)**.\n        - Include **one topic for repository structure named \'Repository Structure\'** and **one for the pipeline or flow named \'Pipeline\'**.\n        - **Strictly return JSON format only.**\n        - **Return a single, properly formatted JSON object.**\n        - Ensure all quotes are properly escaped.\n        - Avoid using newlines within text fields.\n\n        **Example Output:**\n        {\n            "topics": [\n                {\n                    "topic": "main.py",\n                    "summary": "This file serves as the primary entry point of the application. It initializes the main execution loop, loads configuration settings from `config.yaml`, and manages API requests. It calls helper functions from `utils.py` and `database.py` for data handling. The main function defines the initialization of UI components and calls `model.py` for inference if needed."\n                },\n                {\n                    "topic": "config.yaml",\n                    "summary": "Stores all environment configurations, model hyperparameters, and API keys. This file is used across multiple scripts to ensure a standardized setup."\n                }\n            ]\n        }\n        ')
    response = model.start_chat().send_message(text)
    return response.text

@st.cache_data(show_spinner=False)
def ss_pdf_text(text):
    model = genai.GenerativeModel(model_name='gemini-2.0-flash', generation_config=generation_config, system_instruction='\n        Extract **all key topics** from the given PDF and generate **detailed summaries**.\n\n        **Rules:**\n        - Identify and list **all major topics and subtopics**.\n        - Summarize each topic in **8-12 sentences**.\n        - Ensure that no important topic is omitted.\n        - Avoid generic summaries—**provide technical depth and clarity**.\n        - **Strictly return JSON format only.**\n\n        **Example Output:**\n        {\n            "topics": [\n                {\n                    "topic": "Deep Learning",\n                    "summary": "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to learn representations of data. It is widely used in computer vision, natural language processing, and autonomous systems. Popular architectures include CNNs (for image processing), RNNs (for sequential data), and Transformers (for NLP tasks). Challenges include large data requirements and computational costs."\n                },\n                {\n                    "topic": "Gradient Descent",\n                    "summary": "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models. It updates model parameters iteratively by computing gradients. Variants include Stochastic Gradient Descent (SGD), Mini-Batch Gradient Descent, and Adam Optimizer."\n                }\n            ]\n        }\n        ')
    response = model.start_chat().send_message(text)
    return response.text


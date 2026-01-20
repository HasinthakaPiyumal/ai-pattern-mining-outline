# Cluster 4

def generate(topic_data):
    if isinstance(topic_data, str):
        try:
            topic_data = json.loads(topic_data)
        except json.JSONDecodeError:
            st.error('Invalid JSON format in topic data')
            st.cache_data.clear()
            st.stop()
    if topic_data is None:
        st.error('Failed to summarize topics.')
        st.cache_data.clear()
        st.stop()
    i = 0
    if topic_data:
        st.write('#### topics and summaries')
        for topic in topic_data['topics']:
            topic_name = topic['topic']
            summary = topic['summary']
            st.write(f'##### 📌 {topic_name}')
            st.write(f'{summary}')
            with st.spinner(text='generating mindmap', show_time=False):
                relationships = extract_relationships(summary)
                if relationships:
                    time.sleep(1)
                    if topic_name == 'Pipeline':
                        mermaid_code = generate_mermaid_code_pipeline(relationships)
                    else:
                        mermaid_code = generate_mermaid_code(relationships)
                    image = mermaid_to_png(mermaid_code)
                    stmd.st_mermaid(mermaid_code)
            if image:
                st.download_button(label='save as image', data=image, file_name='mindpalace_diagram.png', mime='image/png', key=i)
            st.divider()
            i += 1
    else:
        st.error('No topics detected.')
        st.cache_data.clear()

@st.cache_data(show_spinner=False)
def extract_relationships(topic_text):
    model = genai.GenerativeModel(model_name='gemini-2.0-flash', generation_config=generation_config, system_instruction='\n        Identify **logical relationships** between topics and structure them **hierarchically** for a mind map.\n\n        **Rules:**\n        - Organize topics into **main topics → subtopics → deeper breakdowns**.\n        - Clearly define **why** one topic relates to another (not just that it does).\n        - **Group related concepts** under larger umbrella topics.\n        - Ensure **a natural progression** from general → specific.\n        - Do cover all important points.\n        - Relationships should not be **too obvious** (avoid generic links).\n        - Ensure a good level of interconnection between topics.\n        - Ensure **completeness**—cover all key topics.\n        - **Strictly return JSON format only.**\n\n        **Example Output:**\n        {\n            "relationships": [\n                {"from": "Deep Learning", "to": "Neural Networks", "relationship": "Built upon"},\n                {"from": "Convolutional Neural Networks", "to": "Image Processing", "relationship": "Used for"}\n            ]\n        }\n        ')
    response = model.start_chat().send_message(topic_text)
    return response.text

def mermaid_to_png(mermaid_code: str):
    url = 'https://kroki.io/mermaid/png'
    headers = {'Content-Type': 'text/plain'}
    response = requests.post(url, data=mermaid_code.encode('utf-8'), headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        return None


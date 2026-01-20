# Cluster 54

@st.cache_resource
def get_app():
    app = MCPApp('Streamlit Agent')
    agent = AgentSpec(name='assistant', instruction='You are a helpful AI assistant with access to various tools.', server_names=['filesystem', 'fetch'])
    app.register_agent('assistant', agent)
    return app


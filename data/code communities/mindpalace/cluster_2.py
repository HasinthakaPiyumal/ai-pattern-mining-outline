# Cluster 2

def generate_mermaid_code(relationships_json):
    if isinstance(relationships_json, str):
        try:
            relationships = json.loads(relationships_json)['relationships']
        except json.JSONDecodeError:
            st.error('Invalid JSON format in relationships')
            st.cache_data.clear()
            return None
    else:
        relationships = relationships_json['relationships']
    mermaid_code = "%%{\n  init: {\n    'theme': 'base',\n    'themeVariables': {\n      'primaryColor': '#e66b22',\n      'primaryTextColor': '#efefef',\n      'primaryBorderColor': '#d9d9d9',\n      'lineColor': '#8d8d8d',\n      'secondaryColor': '#212121',\n      'tertiaryColor': '#fff'\n    }\n  }\n}%%\nflowchart LR;\n"
    nodes = {}
    node_counter = 0
    for relation in relationships:
        from_topic = sanitize_text(relation['from'])
        to_topic = sanitize_text(relation['to'])
        relation_text = sanitize_text(relation['relationship'])
        if from_topic not in nodes:
            nodes[from_topic] = get_node_label(node_counter)
            node_counter += 1
        if to_topic not in nodes:
            nodes[to_topic] = get_node_label(node_counter)
            node_counter += 1
        mermaid_code += f'  {nodes[from_topic]}["{from_topic}"] -->|{relation_text}| {nodes[to_topic]}["{to_topic}"];'
    mermaid_code += '\n    classDef customStyle stroke:#e66b22,stroke-width:2px,rx:10px,ry:10px;\n    '
    for node in nodes.values():
        mermaid_code += f'  class {node} customStyle;\n'
        mermaid_code += f'  style {node} font-size:20px;\n'
    return mermaid_code

def sanitize_text(text):
    return re.sub('[^a-zA-Z0-9\\s\\-\\>\\:\\.\\*\\=\\-\\^\\+\\,\\<\\>\\?\\%\\\\]', '', text)

def get_node_label(index):
    """Generate a label like A, B, ..., Z, AA, AB, etc."""
    letters = string.ascii_uppercase
    label = ''
    while index >= 0:
        label = letters[index % 26] + label
        index = index // 26 - 1
    return label

def generate_mermaid_code_pipeline(relationships_json):
    if isinstance(relationships_json, str):
        try:
            relationships = json.loads(relationships_json)['relationships']
        except json.JSONDecodeError:
            st.error('Invalid JSON format in relationships')
            st.cache_data.clear()
            return None
    else:
        relationships = relationships_json['relationships']
    mermaid_code = "%%{\n  init: {\n    'theme': 'base',\n    'themeVariables': {\n      'primaryColor': '#e66b22',\n      'primaryTextColor': '#efefef',\n      'primaryBorderColor': '#d9d9d9',\n      'lineColor': '#8d8d8d',\n      'secondaryColor': '#212121',\n      'tertiaryColor': '#fff'\n    }\n  }\n}%%\nflowchart TD;\n"
    nodes = {}
    node_counter = 0
    for relation in relationships:
        from_topic = sanitize_text(relation['from'])
        to_topic = sanitize_text(relation['to'])
        relation_text = sanitize_text(relation['relationship'])
        if from_topic not in nodes:
            nodes[from_topic] = get_node_label(node_counter)
            node_counter += 1
        if to_topic not in nodes:
            nodes[to_topic] = get_node_label(node_counter)
            node_counter += 1
        mermaid_code += f'  {nodes[from_topic]}["{from_topic}"] -->|{relation_text}| {nodes[to_topic]}["{to_topic}"];'
    mermaid_code += '\n    classDef customStyle stroke:#e66b22,stroke-width:2px,rx:10px,ry:10px;\n    '
    for node in nodes.values():
        mermaid_code += f'  class {node} customStyle;\n'
        mermaid_code += f'  style {node} font-size:20px;\n'
    return mermaid_code


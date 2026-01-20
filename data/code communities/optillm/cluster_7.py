# Cluster 7

def format_question(category: str, question: str, answer: str) -> Dict[str, Any]:
    """Format a question for the benchmark dataset"""
    if not question or not answer:
        raise ValueError(f'Empty question or answer in {category}')
    return {'id': f'{category}_{random.getrandbits(32):08x}', 'category': category, 'question': clean_text(question), 'answer': clean_text(answer), 'metadata': {'source': SOURCES[category]['name'], 'type': category, 'difficulty': 'challenging'}}

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing newlines"""
    return ' '.join(text.replace('\r', '\n').split())


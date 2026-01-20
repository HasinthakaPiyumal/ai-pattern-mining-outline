# Cluster 23

def run(system_prompt: str, initial_query: str, client=None, model: str=None, request_config: Optional[Dict]=None) -> Tuple[str, int]:
    """
    Web search plugin that uses Chrome to search Google and return results
    
    Args:
        system_prompt: System prompt for the conversation
        initial_query: User's query that may contain search requests
        client: OpenAI client (unused for this plugin)
        model: Model name (unused for this plugin) 
        request_config: Optional configuration dict with keys:
            - num_results: Number of search results (default: 10)
            - delay_seconds: Delay between searches in seconds (default: random 4-32)
                            Set to 0 to disable delays, or specify exact seconds
            - headless: Run browser in headless mode (default: False)
            - timeout: Browser timeout in seconds (default: 30)
            - session_manager: BrowserSessionManager instance for session reuse
    
    Returns:
        Tuple of (enhanced_query_with_search_results, completion_tokens)
    """
    config = request_config or {}
    num_results = config.get('num_results', 10)
    delay_seconds = config.get('delay_seconds', None)
    headless = config.get('headless', False)
    timeout = config.get('timeout', 30)
    session_manager = config.get('session_manager', None)
    search_queries = extract_search_queries(initial_query)
    if not search_queries:
        return (initial_query, 0)
    own_session = session_manager is None
    try:
        if own_session:
            searcher = GoogleSearcher(headless=headless, timeout=timeout)
        enhanced_query = initial_query
        for query in search_queries:
            if session_manager:
                results = session_manager.search(query, num_results=num_results, delay_seconds=delay_seconds)
            else:
                results = searcher.search(query, num_results=num_results, delay_seconds=delay_seconds)
            if results:
                formatted_results = format_search_results(query, results)
                enhanced_query = f'{enhanced_query}\n\n[Web Search Results]:\n{formatted_results}'
            else:
                enhanced_query = f"{enhanced_query}\n\n[Web Search Results]:\nNo results found for '{query}'. This may be due to network issues or search restrictions."
        return (enhanced_query, 0)
    except Exception as e:
        error_msg = f'Web search error: {str(e)}'
        enhanced_query = f'{initial_query}\n\n[Web Search Error]: {error_msg}'
        return (enhanced_query, 0)
    finally:
        if own_session and 'searcher' in locals():
            searcher.close()

def extract_search_queries(text: str) -> List[str]:
    """Extract potential search queries from the input text"""
    text = text.strip()
    for prefix in ['User:', 'user:', 'User ', 'user ', 'Assistant:', 'assistant:', 'System:', 'system:']:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    search_patterns = ['search for[:\\s]+(\\S[^\\n]*?)(?:\\s*\\n|$)', 'find information about[:\\s]+(\\S[^\\n]*?)(?:\\s*\\n|$)', 'look up[:\\s]+(\\S[^\\n]*?)(?:\\s*\\n|$)', 'research[:\\s]+(\\S[^\\n]*?)(?:\\s*\\n|$)']
    queries = []
    for pattern in search_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            cleaned = cleaned.rstrip('"\'')
            cleaned = cleaned.lstrip('"\'')
            if cleaned:
                queries.append(cleaned)
    if not queries:
        search_prefixes = ['search for', 'find information about', 'look up', 'research']
        text_lower = text.lower().strip()
        is_empty_search = any((text_lower.startswith(prefix) and len(text_lower.replace(prefix, '').strip().strip('"\'')) < 2 for prefix in search_prefixes))
        if not is_empty_search:
            cleaned_query = text.replace('?', '').strip()
            cleaned_query = cleaned_query.strip('"\'')
            if cleaned_query and len(cleaned_query.split()) > 2:
                queries.append(cleaned_query)
            else:
                cleaned_query = re.sub('[^\\w\\s\\.]', ' ', text)
                cleaned_query = ' '.join(cleaned_query.split())
                cleaned_query = cleaned_query.strip('"\'')
                if len(cleaned_query) > 100:
                    cleaned_query = cleaned_query[:100].rsplit(' ', 1)[0]
                if cleaned_query and len(cleaned_query) > 2:
                    queries.append(cleaned_query)
    return queries

def format_search_results(query: str, results: List[Dict[str, str]]) -> str:
    """Format search results into readable text"""
    if not results:
        return f'No search results found for: {query}'
    formatted = f"Search results for '{query}':\n\n"
    for i, result in enumerate(results, 1):
        formatted += f'{i}. **{result['title']}**\n'
        formatted += f'   URL: {result['url']}\n'
        if result['snippet']:
            formatted += f'   Summary: {result['snippet']}\n'
        formatted += '\n'
    return formatted


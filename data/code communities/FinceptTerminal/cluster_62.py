# Cluster 62

def make_request(endpoint, params=None):
    """Efficient request function with counter"""
    global request_count
    if params is None:
        params = {}
    params['apikey'] = API_KEY
    try:
        response = requests.get(f'{BASE_URL}/{endpoint}', params=params)
        request_count += 1
        print(f'📡 Request {request_count}: {endpoint}')
        if response.status_code == 200:
            return response.json()
        else:
            print(f'❌ Error {response.status_code}: {endpoint}')
            return None
    except Exception as e:
        print(f'❌ Exception: {e}')
        return None

def safe_format(value, format_type='number', decimal_places=2):
    """Safely format values to avoid errors"""
    if value is None or value == 'N/A' or value == '':
        return 'N/A'
    try:
        if format_type == 'percentage':
            return f'{float(value):.{decimal_places}%}'
        elif format_type == 'number':
            return f'{float(value):,.{decimal_places}f}'
        elif format_type == 'currency':
            return f'${float(value):,.0f}'
        else:
            return str(value)
    except:
        return str(value) if value is not None else 'N/A'

def log_result(section, content):
    """Log results for TXT file output"""
    analysis_results.append(f'\n{section}')
    analysis_results.append('=' * len(section))
    analysis_results.append(content)

def get_schema_filelist(query: str='', url: str='', use_cache: bool=True) -> List:
    """Get a list of schema files from the SEC website."""
    results: List = []
    url = url if url else f'https://xbrl.fasb.org/us-gaap/{query}'
    _url = url
    _url = url + '/' if query else _url
    try:
        response = make_request(_url)
        data = read_html(response.content)[0]['Name'].dropna()
        if len(data) > 0:
            data.iloc[0] = url if not query else url + '/'
            results = data.to_list()
    except Exception:
        pass
    return results


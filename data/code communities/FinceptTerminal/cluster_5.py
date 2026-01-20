# Cluster 5

def ping() -> Dict[str, Any]:
    return _make_request('ping')

def _make_request(endpoint: str, params: Optional[Dict[str, Any]]=None) -> Any:
    """A private helper function to handle all API requests and errors."""
    full_url = f'{BASE_URL}/{endpoint}'
    if API_KEY:
        if params is None:
            params = {}
        params['x_cg_pro_api_key'] = API_KEY
    try:
        response = requests.get(full_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {'error': f'HTTP Error: {e.response.status_code} - {e.response.text}'}
    except requests.exceptions.RequestException as e:
        return {'error': f'Network or request error: {str(e)}'}
    except json.JSONDecodeError:
        return {'error': 'Failed to decode API response.'}

def get_simple_price(ids: str, vs_currencies: str) -> Dict[str, Any]:
    params = {'ids': ids, 'vs_currencies': vs_currencies, 'include_market_cap': 'true', 'include_24hr_vol': 'true', 'include_24hr_change': 'true', 'include_last_updated_at': 'true'}
    return _make_request('simple/price', params=params)

def get_token_price(platform_id: str, contract_addresses: str, vs_currencies: str) -> Dict[str, Any]:
    params = {'contract_addresses': contract_addresses, 'vs_currencies': vs_currencies}
    return _make_request(f'simple/token_price/{platform_id}', params=params)

def get_supported_vs_currencies() -> List[str]:
    return _make_request('simple/supported_vs_currencies')

def get_coin_list(include_platform: bool=False) -> List[Dict[str, Any]]:
    return _make_request('coins/list', params={'include_platform': str(include_platform).lower()})

def get_coin_markets(vs_currency: str, **kwargs) -> List[Dict[str, Any]]:
    params = {'vs_currency': vs_currency, **kwargs}
    return _make_request('coins/markets', params=params)

def get_coin_details(coin_id: str) -> Dict[str, Any]:
    return _make_request(f'coins/{coin_id}')

def get_coin_tickers(coin_id: str) -> Dict[str, Any]:
    return _make_request(f'coins/{coin_id}/tickers')

def get_coin_history(coin_id: str, date: str) -> Dict[str, Any]:
    return _make_request(f'coins/{coin_id}/history', params={'date': date})

def get_market_chart(coin_id: str, vs_currency: str, days: str) -> Dict[str, Any]:
    return _make_request(f'coins/{coin_id}/market_chart', params={'vs_currency': vs_currency, 'days': days})

def get_market_chart_range(coin_id: str, vs_currency: str, from_unix: str, to_unix: str) -> Dict[str, Any]:
    return _make_request(f'coins/{coin_id}/market_chart/range', params={'vs_currency': vs_currency, 'from': from_unix, 'to': to_unix})

def get_coin_ohlc(coin_id: str, vs_currency: str, days: str) -> List[Any]:
    return _make_request(f'coins/{coin_id}/ohlc', params={'vs_currency': vs_currency, 'days': days})

def get_contract_info(platform_id: str, contract_address: str) -> Dict[str, Any]:
    return _make_request(f'coins/{platform_id}/contract/{contract_address}')

def get_contract_market_chart(platform_id: str, contract_address: str, vs_currency: str, days: str) -> Dict[str, Any]:
    return _make_request(f'coins/{platform_id}/contract/{contract_address}/market_chart', params={'vs_currency': vs_currency, 'days': days})

def get_contract_market_chart_range(platform_id: str, contract_address: str, vs_currency: str, from_unix: str, to_unix: str) -> Dict[str, Any]:
    return _make_request(f'coins/{platform_id}/contract/{contract_address}/market_chart/range', params={'vs_currency': vs_currency, 'from': from_unix, 'to': to_unix})

def get_asset_platforms() -> List[Dict[str, Any]]:
    return _make_request('asset_platforms')

def get_categories_list() -> List[Dict[str, Any]]:
    return _make_request('coins/categories/list')

def get_categories_with_market_data() -> List[Dict[str, Any]]:
    return _make_request('coins/categories')

def get_exchange_list() -> List[Dict[str, Any]]:
    return _make_request('exchanges')

def get_exchange_id_name_list() -> List[Dict[str, Any]]:
    return _make_request('exchanges/list')

def get_exchange_details(exchange_id: str) -> Dict[str, Any]:
    return _make_request(f'exchanges/{exchange_id}')

def get_exchange_tickers(exchange_id: str) -> Dict[str, Any]:
    return _make_request(f'exchanges/{exchange_id}/tickers')

def get_exchange_volume_chart(exchange_id: str, days: str) -> List[Any]:
    return _make_request(f'exchanges/{exchange_id}/volume_chart', params={'days': days})

def get_indexes_list() -> List[Dict[str, Any]]:
    return _make_request('indexes')

def get_index_details(market_id: str, index_id: str) -> Dict[str, Any]:
    return _make_request(f'indexes/{market_id}/{index_id}')

def get_index_list_by_market() -> List[Dict[str, Any]]:
    return _make_request('indexes/list')

def get_derivatives_list() -> List[Dict[str, Any]]:
    return _make_request('derivatives')

def get_derivatives_exchanges() -> List[Dict[str, Any]]:
    return _make_request('derivatives/exchanges')

def get_derivatives_exchange_details(exchange_id: str) -> Dict[str, Any]:
    return _make_request(f'derivatives/exchanges/{exchange_id}', params={'include_tickers': 'all'})

def get_derivatives_exchange_list() -> List[Dict[str, Any]]:
    return _make_request('derivatives/exchanges/list')

def get_nft_list() -> List[Dict[str, Any]]:
    return _make_request('nfts/list')

def get_nft_details(nft_id: str) -> Dict[str, Any]:
    return _make_request(f'nfts/{nft_id}')

def get_nft_contract_info(platform_id: str, contract_address: str) -> Dict[str, Any]:
    return _make_request(f'nfts/{platform_id}/contract/{contract_address}')

def get_exchange_rates() -> Dict[str, Any]:
    return _make_request('exchange_rates')

def get_search(query: str) -> Dict[str, Any]:
    return _make_request('search', params={'query': query})

def get_trending_coins() -> List[Dict[str, Any]]:
    return _make_request('search/trending')

def get_global_data() -> Dict[str, Any]:
    return _make_request('global')

def get_global_defi_data() -> Dict[str, Any]:
    return _make_request('global/decentralized_finance_defi')

def get_company_treasury(coin_id: str) -> Dict[str, Any]:
    return _make_request(f'companies/public_treasury/{coin_id}')

def main():
    """Main CLI entry point. This is a simplified router."""
    if len(sys.argv) < 2:
        print('Usage: python coingecko_complete_wrapper.py <command> [args...]')
        return
    cmd = sys.argv[1]
    args = sys.argv[2:]
    result = {}
    try:
        if cmd == 'ping':
            result = ping()
        elif cmd == 'price':
            result = get_simple_price(ids=args[0], vs_currencies=args[1])
        elif cmd == 'coin-list':
            result = get_coin_list()
        elif cmd == 'details':
            result = get_coin_details(coin_id=args[0])
        elif cmd == 'history':
            result = get_coin_history(coin_id=args[0], date=args[1])
        elif cmd == 'exchanges':
            result = get_exchange_list()
        elif cmd == 'trending':
            result = get_trending_coins()
        elif cmd == 'global':
            result = get_global_data()
        elif cmd == 'nft-list':
            result = get_nft_list()
        elif cmd == 'nft-details':
            result = get_nft_details(nft_id=args[0])
        elif cmd == 'treasury':
            result = get_company_treasury(coin_id=args[0])
        else:
            result = {'error': f'Unknown or unimplemented command: {cmd}'}
    except IndexError:
        result = {'error': f"Missing arguments for command '{cmd}'."}
    except Exception as e:
        result = {'error': f'An unexpected error occurred: {e}'}
    print(json.dumps(result, indent=2))

def test_connection() -> Dict[str, Any]:
    """Test API connection with a simple call."""
    return _make_request('credit-ratings')

def get_credit_ratings(country: str=None) -> Dict[str, Any]:
    """1. Get sovereign credit ratings for a specific country or all countries."""
    if country:
        endpoint = f'credit-ratings/country/{country}'
    else:
        endpoint = 'credit-ratings'
    data = _make_request(endpoint)
    if data and 'error' not in data and isinstance(data, list):
        return {'data': data, 'count': len(data), 'source': 'Trading Economics', 'timestamp': datetime.now().isoformat()}
    return data

def get_credit_ratings_by_agency(agency: str) -> Dict[str, Any]:
    """2. Get credit ratings filtered by rating agency (S&P, Moody's, Fitch)."""
    data = _make_request('credit-ratings', {'agency': agency})
    if data and 'error' not in data and isinstance(data, list):
        return {'data': data, 'agency': agency, 'count': len(data), 'timestamp': datetime.now().isoformat()}
    return data

def get_historical_credit_ratings(country: str, start_date: str=None, end_date: str=None) -> Dict[str, Any]:
    """3. Get historical credit ratings for a country."""
    params = {}
    if start_date:
        params['startDate'] = start_date
    if end_date:
        params['endDate'] = end_date
    endpoint = f'credit-ratings/country/{country}/historical'
    data = _make_request(endpoint, params)
    if data and 'error' not in data:
        return {'data': data, 'country': country, 'period': {'start': start_date, 'end': end_date}, 'timestamp': datetime.now().isoformat()}
    return data

def get_rating_changes() -> Dict[str, Any]:
    """4. Get recent credit rating changes (upgrades/downgrades)."""
    data = _make_request('credit-ratings/changes')
    if data and 'error' not in data:
        return {'data': data, 'timestamp': datetime.now().isoformat()}
    return data

def get_government_bond_yields(country: str=None) -> Dict[str, Any]:
    """5. Get government bond yields for a specific country or all countries."""
    if country:
        endpoint = f'country/{country}/bond-yield'
    else:
        endpoint = 'markets/bond'
    data = _make_request(endpoint)
    if data and 'error' not in data and isinstance(data, list):
        return {'data': data, 'country': country if country else 'All', 'count': len(data), 'timestamp': datetime.now().isoformat()}
    return data

def get_bond_symbol(symbol: str) -> Dict[str, Any]:
    """6. Get specific government bond data by symbol (e.g., US10Y, UK10Y)."""
    data = _make_request(f'markets/symbol/{symbol}')
    if data and 'error' not in data and isinstance(data, list) and (len(data) > 0):
        return {'data': data[0], 'symbol': symbol, 'timestamp': datetime.now().isoformat()}
    elif isinstance(data, list) and len(data) == 0:
        return {'error': f'No data found for symbol: {symbol}'}
    return data

def get_us_treasury_yields() -> Dict[str, Any]:
    """7. Get US Treasury yields for all durations."""
    data = _make_request('markets/bond', {'country': 'united states'})
    if data and 'error' not in data and isinstance(data, list):
        return {'data': data, 'country': 'United States', 'count': len(data), 'timestamp': datetime.now().isoformat()}
    return data

def get_european_bond_yields() -> Dict[str, Any]:
    """8. Get European government bond yields."""
    data = _make_request('markets/bond', {'country': 'european union'})
    if data and 'error' not in data and isinstance(data, list):
        return {'data': data, 'region': 'European Union', 'count': len(data), 'timestamp': datetime.now().isoformat()}
    return data

def get_historical_bond_yields(symbol: str, start_date: str=None, end_date: str=None) -> Dict[str, Any]:
    """9. Get historical bond yields for a specific symbol."""
    params = {}
    if start_date:
        params['startDate'] = start_date
    if end_date:
        params['endDate'] = end_date
    endpoint = f'markets/symbol/{symbol}/historical'
    data = _make_request(endpoint, params)
    if data and 'error' not in data:
        return {'data': data, 'symbol': symbol, 'period': {'start': start_date, 'end': end_date}, 'timestamp': datetime.now().isoformat()}
    return data

def get_yield_curve(country: str) -> Dict[str, Any]:
    """10. Get yield curve data for a country."""
    data = _make_request(f'country/{country}/yield-curve')
    if data and 'error' not in data:
        return {'data': data, 'country': country, 'timestamp': datetime.now().isoformat()}
    return data

def get_country_financial_data(country: str) -> Dict[str, Any]:
    """11. Get comprehensive financial data (credit ratings + bond yields) for a country."""
    try:
        credit_data = get_credit_ratings(country)
        bond_data = get_government_bond_yields(country)
        result = {'country': country, 'timestamp': datetime.now().isoformat(), 'credit_ratings': credit_data, 'bond_yields': bond_data}
        credit_success = credit_data and 'error' not in credit_data
        bond_success = bond_data and 'error' not in bond_data
        result['summary'] = {'credit_ratings_available': credit_success, 'bond_yields_available': bond_success, 'overall_success': credit_success or bond_success}
        return result
    except Exception as e:
        return {'error': f'Failed to get comprehensive data for {country}: {str(e)}'}

def get_global_summary() -> Dict[str, Any]:
    """12. Get global summary of credit ratings and bond markets."""
    try:
        ratings_data = _make_request('credit-ratings')
        if ratings_data and 'error' not in ratings_data and isinstance(ratings_data, list):
            agencies = list(set((rating.get('Agency', 'Unknown') for rating in ratings_data)))
            countries = list(set((rating.get('Country', 'Unknown') for rating in ratings_data)))
            rating_distribution = {}
            for rating in ratings_data:
                rating_val = rating.get('Rating', 'Unknown')
                rating_distribution[rating_val] = rating_distribution.get(rating_val, 0) + 1
            ratings_summary = {'total_ratings': len(ratings_data), 'agencies': agencies, 'countries_count': len(countries), 'rating_distribution': rating_distribution}
        else:
            ratings_summary = {'error': 'Failed to fetch ratings data'}
        bond_data = _make_request('markets/bond-overview')
        if bond_data and 'error' not in bond_data:
            bond_summary = bond_data
        else:
            bond_summary = {'error': 'Failed to fetch bond overview'}
        return {'timestamp': datetime.now().isoformat(), 'credit_ratings_summary': ratings_summary, 'bond_markets_summary': bond_summary, 'data_source': 'Trading Economics'}
    except Exception as e:
        return {'error': f'Failed to generate global summary: {str(e)}'}

def search_bonds(query: str) -> Dict[str, Any]:
    """13. Search for bond symbols and instruments."""
    data = _make_request('search/bond', {'q': query})
    if data and 'error' not in data:
        return {'data': data, 'query': query, 'timestamp': datetime.now().isoformat()}
    return data

def get_supported_countries() -> Dict[str, Any]:
    """14. Get list of supported countries for credit ratings and bonds."""
    ratings_data = _make_request('credit-ratings')
    if ratings_data and 'error' not in ratings_data and isinstance(ratings_data, list):
        countries = sorted(list(set((rating.get('Country', 'Unknown') for rating in ratings_data))))
        return {'data': countries, 'count': len(countries), 'timestamp': datetime.now().isoformat()}
    return ratings_data

def get_rating_calendar() -> Dict[str, Any]:
    """15. Get calendar of upcoming rating events."""
    data = _make_request('calendar/ratings')
    if data and 'error' not in data:
        return {'data': data, 'timestamp': datetime.now().isoformat()}
    return data

def main():
    """Main Command-Line Interface entry point."""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: python trading_economics_data.py <command> [args]', 'commands': ['test - Test API connection', 'ratings [country] - Get credit ratings (optional country)', "ratings_by_agency <agency> - Get ratings by agency (S&P, Moody's, Fitch)", 'ratings_history <country> [start_date] [end_date] - Historical ratings', 'rating_changes - Get recent rating changes', 'bonds [country] - Get bond yields (optional country)', 'bond <symbol> - Get specific bond data (e.g., US10Y)', 'us_treasuries - Get US Treasury yields', 'european_bonds - Get European bond yields', 'bond_history <symbol> [start_date] [end_date] - Historical bond yields', 'yield_curve <country> - Get yield curve data', 'country_data <country> - Get comprehensive country data', 'global_summary - Get global market summary', 'search_bonds <query> - Search for bonds', 'countries - Get supported countries', 'calendar - Get rating calendar'], 'examples': ['python trading_economics_data.py test', 'python trading_economics_data.py ratings Sweden', 'python trading_economics_data.py bond US10Y', 'python trading_economics_data.py country_data United States', 'python trading_economics_data.py ratings_history Sweden 2023-01-01 2023-12-31']}, indent=2))
        sys.exit(1)
    command = sys.argv[1].lower()
    result = {}
    try:
        if command == 'test':
            result = test_connection()
        elif command == 'ratings':
            country = sys.argv[2] if len(sys.argv) > 2 else None
            result = get_credit_ratings(country)
        elif command == 'ratings_by_agency':
            agency = sys.argv[2]
            result = get_credit_ratings_by_agency(agency)
        elif command == 'ratings_history':
            country = sys.argv[2]
            start_date = sys.argv[3] if len(sys.argv) > 3 else None
            end_date = sys.argv[4] if len(sys.argv) > 4 else None
            result = get_historical_credit_ratings(country, start_date, end_date)
        elif command == 'rating_changes':
            result = get_rating_changes()
        elif command == 'bonds':
            country = sys.argv[2] if len(sys.argv) > 2 else None
            result = get_government_bond_yields(country)
        elif command == 'bond':
            symbol = sys.argv[2]
            result = get_bond_symbol(symbol)
        elif command == 'us_treasuries':
            result = get_us_treasury_yields()
        elif command == 'european_bonds':
            result = get_european_bond_yields()
        elif command == 'bond_history':
            symbol = sys.argv[2]
            start_date = sys.argv[3] if len(sys.argv) > 3 else None
            end_date = sys.argv[4] if len(sys.argv) > 4 else None
            result = get_historical_bond_yields(symbol, start_date, end_date)
        elif command == 'yield_curve':
            country = sys.argv[2]
            result = get_yield_curve(country)
        elif command == 'country_data':
            country = ' '.join(sys.argv[2:])
            result = get_country_financial_data(country)
        elif command == 'global_summary':
            result = get_global_summary()
        elif command == 'search_bonds':
            query = ' '.join(sys.argv[2:])
            result = search_bonds(query)
        elif command == 'countries':
            result = get_supported_countries()
        elif command == 'calendar':
            result = get_rating_calendar()
        else:
            result = {'error': f'Unknown command: {command}'}
    except IndexError:
        result = {'error': 'Missing required arguments for command.'}
    except Exception as e:
        result = {'error': f'An unexpected error occurred: {str(e)}'}
    print(json.dumps(result, indent=2))

def get_dataset_details(dataset_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific dataset

    Args:
        dataset_id: The unique ID or name of the dataset

    Returns:
        JSON response with dataset details
    """
    try:
        params = {'id': dataset_id}
        result = _make_request('package_show', params)
        if result['error']:
            return result
        dataset_data = result.get('data', {})
        enhanced_dataset = {'id': dataset_data.get('id'), 'name': dataset_data.get('name'), 'title': dataset_data.get('title'), 'notes': dataset_data.get('notes', ''), 'url': dataset_data.get('url'), 'author': dataset_data.get('author'), 'author_email': dataset_data.get('author_email'), 'maintainer': dataset_data.get('maintainer'), 'maintainer_email': dataset_data.get('maintainer_email'), 'license_id': dataset_data.get('license_id'), 'license_title': dataset_data.get('license_title'), 'organization': dataset_data.get('organization', {}).get('name') if dataset_data.get('organization') else None, 'metadata_created': dataset_data.get('metadata_created'), 'metadata_modified': dataset_data.get('metadata_modified'), 'state': dataset_data.get('state'), 'version': dataset_data.get('version'), 'tags': [tag.get('display_name') for tag in dataset_data.get('tags', [])]}
        result['data'] = enhanced_dataset
        result['metadata']['dataset_id'] = dataset_id
        result['metadata']['description'] = f'Details for dataset {dataset_id}'
        return result
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Error fetching dataset details: {str(e)}'}

def get_dataset_resources(dataset_id: str) -> Dict[str, Any]:
    """
    Get all data files (resources) for a specific dataset

    Args:
        dataset_id: The unique ID or name of the dataset

    Returns:
        JSON response with resource list
    """
    try:
        params = {'id': dataset_id}
        result = _make_request('package_show', params)
        if result['error']:
            return result
        dataset_data = result.get('data', {})
        resources = dataset_data.get('resources', [])
        enhanced_data = []
        for resource in resources:
            enhanced_resource = {'id': resource.get('id'), 'name': resource.get('name'), 'description': resource.get('description', ''), 'format': resource.get('format', ''), 'url': resource.get('url', ''), 'size': resource.get('size'), 'mimetype': resource.get('mimetype'), 'mimetype_inner': resource.get('mimetype_inner'), 'created': resource.get('created'), 'last_modified': resource.get('last_modified'), 'resource_type': resource.get('resource_type'), 'package_id': dataset_id, 'position': resource.get('position'), 'cache_last_updated': resource.get('cache_last_updated'), 'webstore_last_updated': resource.get('webstore_last_updated')}
            enhanced_data.append(enhanced_resource)
        result['data'] = enhanced_data
        result['metadata']['dataset_id'] = dataset_id
        result['metadata']['dataset_name'] = dataset_data.get('name')
        result['metadata']['resource_count'] = len(enhanced_data)
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error fetching dataset resources: {str(e)}'}

def get_resource_info(resource_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific resource

    Args:
        resource_id: The unique ID of the resource

    Returns:
        JSON response with resource details
    """
    try:
        params = {'id': resource_id}
        result = _make_request('resource_show', params)
        if result['error']:
            return result
        resource_data = result.get('data', {})
        enhanced_resource = {'id': resource_data.get('id'), 'name': resource_data.get('name'), 'description': resource_data.get('description', ''), 'format': resource_data.get('format', ''), 'url': resource_data.get('url', ''), 'size': resource_data.get('size'), 'mimetype': resource_data.get('mimetype'), 'mimetype_inner': resource_data.get('mimetype_inner'), 'created': resource_data.get('created'), 'last_modified': resource_data.get('last_modified'), 'resource_type': resource_data.get('resource_type'), 'package_id': resource_data.get('package_id'), 'position': resource_data.get('position'), 'cache_last_updated': resource_data.get('cache_last_updated'), 'webstore_last_updated': resource_data.get('webstore_last_updated')}
        result['data'] = enhanced_resource
        result['metadata']['resource_id'] = resource_id
        return result
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Error fetching resource info: {str(e)}'}

def search_datasets(query: str, rows: int=50) -> Dict[str, Any]:
    """
    Search for datasets across all organizations

    Args:
        query: Search query string
        rows: Number of results to return (default: 50)

    Returns:
        JSON response with search results
    """
    try:
        params = {'q': query, 'rows': rows}
        result = _make_request('package_search', params)
        if result['error']:
            return result
        search_data = result.get('data', {})
        datasets = search_data.get('results', [])
        enhanced_data = []
        for dataset in datasets:
            enhanced_dataset = {'id': dataset.get('id'), 'name': dataset.get('name'), 'title': dataset.get('title'), 'notes': dataset.get('notes', ''), 'organization': dataset.get('organization', {}).get('name') if dataset.get('organization') else None, 'metadata_created': dataset.get('metadata_created'), 'metadata_modified': dataset.get('metadata_modified'), 'num_resources': len(dataset.get('resources', [])), 'tags': [tag.get('display_name') for tag in dataset.get('tags', [])]}
            enhanced_data.append(enhanced_dataset)
        result['data'] = enhanced_data
        result['metadata']['query'] = query
        result['metadata']['total_count'] = search_data.get('count', 0)
        result['metadata']['returned_count'] = len(enhanced_data)
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error searching datasets: {str(e)}'}

def get_dataset_details(dataset_id: str) -> Dict[str, Any]:
    """Get details for a specific dataset including its resources/distributions"""
    if not dataset_id:
        return {'error': 'Dataset ID is required'}
    result = _make_request(f'catalog/dataset/{urllib.parse.quote(dataset_id)}')
    if result.get('error') is None and result.get('data', {}).get('items'):
        dataset = result['data']['items'][0]
        distributions = dataset.get('distribution', [])
        format_counts = {}
        for dist in distributions:
            format_field = dist.get('format', '')
            if isinstance(format_field, str):
                if format_field.startswith('http'):
                    format_label = format_field.split('/')[-1].upper()
                else:
                    format_label = format_field
            elif isinstance(format_field, dict):
                format_label = format_field.get('label', 'Unknown')
            else:
                format_label = 'Unknown'
            format_counts[format_label] = format_counts.get(format_label, 0) + 1
        result['metadata']['distribution_count'] = len(distributions)
        result['metadata']['format_summary'] = format_counts
        result['metadata']['dataset_id'] = dataset_id
        result['metadata']['description'] = f'Dataset details with {len(distributions)} data files'
    return result

def get_publishers() -> Dict[str, Any]:
    """
    Get list of all data publishers (organizations) in data.gov.uk

    Returns:
        JSON response with publisher list
    """
    try:
        result = _make_request('organization_list')
        if result['error']:
            return result
        enhanced_data = []
        publishers = result.get('data', [])
        for publisher_id in publishers:
            enhanced_publisher = {'id': publisher_id, 'name': publisher_id.replace('-', ' ').title(), 'display_name': publisher_id.replace('-', ' ').title()}
            enhanced_data.append(enhanced_publisher)
        result['data'] = enhanced_data
        result['metadata']['count'] = len(enhanced_data)
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error fetching publishers: {str(e)}'}

def get_publisher_details(publisher_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific publisher

    Args:
        publisher_id: The unique ID of the publisher

    Returns:
        JSON response with publisher details
    """
    try:
        params = {'id': publisher_id}
        result = _make_request('organization_show', params)
        if result['error']:
            return result
        publisher_data = result.get('data', {})
        enhanced_publisher = {'id': publisher_data.get('id'), 'name': publisher_data.get('name'), 'title': publisher_data.get('title'), 'description': publisher_data.get('description'), 'image_url': publisher_data.get('image_display_url'), 'created': publisher_data.get('created'), 'num_datasets': publisher_data.get('package_count', 0), 'users': publisher_data.get('users', [])}
        result['data'] = enhanced_publisher
        result['metadata']['publisher_id'] = publisher_id
        return result
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Error fetching publisher details: {str(e)}'}

def get_datasets_by_publisher(publisher_id: str, rows: int=100) -> Dict[str, Any]:
    """
    Get all datasets published by a specific publisher

    Args:
        publisher_id: The unique ID of the publisher
        rows: Number of datasets to return (default: 100)

    Returns:
        JSON response with dataset list
    """
    try:
        query = f'owner_org:{publisher_id}'
        params = {'q': query, 'rows': rows}
        result = _make_request('package_search', params)
        if result['error']:
            return result
        search_data = result.get('data', {})
        datasets = search_data.get('results', [])
        enhanced_data = []
        for dataset in datasets:
            enhanced_dataset = {'id': dataset.get('id'), 'name': dataset.get('name'), 'title': dataset.get('title'), 'notes': dataset.get('notes', ''), 'publisher_id': publisher_id, 'metadata_created': dataset.get('metadata_created'), 'metadata_modified': dataset.get('metadata_modified'), 'state': dataset.get('state'), 'num_resources': len(dataset.get('resources', [])), 'tags': [tag.get('display_name') for tag in dataset.get('tags', [])]}
            enhanced_data.append(enhanced_dataset)
        result['data'] = enhanced_data
        result['metadata']['publisher_id'] = publisher_id
        result['metadata']['total_count'] = search_data.get('count', 0)
        result['metadata']['returned_count'] = len(enhanced_data)
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error fetching datasets: {str(e)}'}

def search_datasets(query: str, rows: int=50) -> Dict[str, Any]:
    """
    Search for datasets across all publishers

    Args:
        query: Search query string
        rows: Number of results to return (default: 50)

    Returns:
        JSON response with search results
    """
    try:
        params = {'q': query, 'rows': rows}
        result = _make_request('package_search', params)
        if result['error']:
            return result
        search_data = result.get('data', {})
        datasets = search_data.get('results', [])
        enhanced_data = []
        for dataset in datasets:
            enhanced_dataset = {'id': dataset.get('id'), 'name': dataset.get('name'), 'title': dataset.get('title'), 'notes': dataset.get('notes', ''), 'organization': dataset.get('organization', {}).get('name') if dataset.get('organization') else None, 'metadata_created': dataset.get('metadata_created'), 'metadata_modified': dataset.get('metadata_modified'), 'num_resources': len(dataset.get('resources', [])), 'tags': [tag.get('display_name') for tag in dataset.get('tags', [])]}
            enhanced_data.append(enhanced_dataset)
        result['data'] = enhanced_data
        result['metadata']['query'] = query
        result['metadata']['total_count'] = search_data.get('count', 0)
        result['metadata']['returned_count'] = len(enhanced_data)
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error searching datasets: {str(e)}'}

def get_dataset_details(dataset_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific dataset

    Args:
        dataset_id: The unique ID or name of the dataset

    Returns:
        JSON response with dataset details
    """
    try:
        params = {'id': dataset_id}
        result = _make_request('package_show', params)
        if result['error']:
            return result
        dataset_data = result.get('data', {})
        enhanced_dataset = {'id': dataset_data.get('id'), 'name': dataset_data.get('name'), 'title': dataset_data.get('title'), 'notes': dataset_data.get('notes', ''), 'url': dataset_data.get('url'), 'author': dataset_data.get('author'), 'author_email': dataset_data.get('author_email'), 'maintainer': dataset_data.get('maintainer'), 'maintainer_email': dataset_data.get('maintainer_email'), 'license_id': dataset_data.get('license_id'), 'license_title': dataset_data.get('license_title'), 'organization': dataset_data.get('organization', {}).get('name') if dataset_data.get('organization') else None, 'metadata_created': dataset_data.get('metadata_created'), 'metadata_modified': dataset_data.get('metadata_modified'), 'state': dataset_data.get('state'), 'version': dataset_data.get('version'), 'tags': [tag.get('display_name') for tag in dataset_data.get('tags', [])]}
        result['data'] = enhanced_dataset
        result['metadata']['dataset_id'] = dataset_id
        return result
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Error fetching dataset details: {str(e)}'}

def get_dataset_resources(dataset_id: str) -> Dict[str, Any]:
    """
    Get all data files (resources) for a specific dataset

    Args:
        dataset_id: The unique ID or name of the dataset

    Returns:
        JSON response with resource list
    """
    try:
        params = {'id': dataset_id}
        result = _make_request('package_show', params)
        if result['error']:
            return result
        dataset_data = result.get('data', {})
        resources = dataset_data.get('resources', [])
        enhanced_data = []
        for resource in resources:
            enhanced_resource = {'id': resource.get('id'), 'name': resource.get('name'), 'description': resource.get('description', ''), 'format': resource.get('format', ''), 'url': resource.get('url', ''), 'size': resource.get('size'), 'mimetype': resource.get('mimetype'), 'mimetype_inner': resource.get('mimetype_inner'), 'created': resource.get('created'), 'last_modified': resource.get('last_modified'), 'resource_type': resource.get('resource_type'), 'package_id': dataset_id, 'position': resource.get('position'), 'cache_last_updated': resource.get('cache_last_updated'), 'webstore_last_updated': resource.get('webstore_last_updated')}
            enhanced_data.append(enhanced_resource)
        result['data'] = enhanced_data
        result['metadata']['dataset_id'] = dataset_id
        result['metadata']['dataset_name'] = dataset_data.get('name')
        result['metadata']['resource_count'] = len(enhanced_data)
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error fetching dataset resources: {str(e)}'}

def get_resource_info(resource_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific resource

    Args:
        resource_id: The unique ID of the resource

    Returns:
        JSON response with resource details
    """
    try:
        params = {'id': resource_id}
        result = _make_request('resource_show', params)
        if result['error']:
            return result
        resource_data = result.get('data', {})
        enhanced_resource = {'id': resource_data.get('id'), 'name': resource_data.get('name'), 'description': resource_data.get('description', ''), 'format': resource_data.get('format', ''), 'url': resource_data.get('url', ''), 'size': resource_data.get('size'), 'mimetype': resource_data.get('mimetype'), 'mimetype_inner': resource_data.get('mimetype_inner'), 'created': resource_data.get('created'), 'last_modified': resource_data.get('last_modified'), 'resource_type': resource_data.get('resource_type'), 'package_id': resource_data.get('package_id'), 'position': resource_data.get('position'), 'cache_last_updated': resource_data.get('cache_last_updated'), 'webstore_last_updated': resource_data.get('webstore_last_updated')}
        result['data'] = enhanced_resource
        result['metadata']['resource_id'] = resource_id
        return result
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Error fetching resource info: {str(e)}'}

def get_recent_datasets(limit: int=50) -> Dict[str, Any]:
    """
    Get recently updated datasets

    Args:
        limit: Maximum number of datasets to return

    Returns:
        JSON response with recent datasets
    """
    try:
        params = {'rows': limit, 'sort': 'metadata_modified desc'}
        result = _make_request('package_search', params)
        if result['error']:
            return result
        search_data = result.get('data', {})
        datasets = search_data.get('results', [])
        enhanced_data = []
        for dataset in datasets:
            enhanced_dataset = {'id': dataset.get('id'), 'name': dataset.get('name'), 'title': dataset.get('title'), 'notes': dataset.get('notes', ''), 'organization': dataset.get('organization', {}).get('name') if dataset.get('organization') else None, 'metadata_modified': dataset.get('metadata_modified'), 'num_resources': len(dataset.get('resources', [])), 'tags': [tag.get('display_name') for tag in dataset.get('tags', [])]}
            enhanced_data.append(enhanced_dataset)
        result['data'] = enhanced_data
        result['metadata']['total_count'] = search_data.get('count', 0)
        result['metadata']['returned_count'] = len(enhanced_data)
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error fetching recent datasets: {str(e)}'}


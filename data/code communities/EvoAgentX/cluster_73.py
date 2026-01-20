# Cluster 73

def rapidapi_test() -> None:
    print('\n===== SINGLE REAL CALL: OpenWeatherMap (extracted) =====\n')
    api_key = os.getenv('RAPIDAPI_KEY')
    if not api_key or api_key.strip().lower() in {'', 'your-api-key'}:
        print('Skipping real call: set RAPIDAPI_KEY to run this test.')
        return
    rapidapi_host = 'open-weather13.p.rapidapi.com'
    toolkit = create_rapidapi_toolkit(schema_path_or_dict=weather_api_spec, rapidapi_key=api_key, rapidapi_host=rapidapi_host, service_name='Open Weather13')
    print('____________ Executing city weather querying ____________')
    city_weather_tool = toolkit.get_tools()[0]
    example_query = {'city': 'new york'}
    print('Qeury inputs: \n', example_query)
    result = city_weather_tool(**example_query)
    print('Query result: \n', result)

def create_rapidapi_toolkit(schema_path_or_dict: Union[str, Dict[str, Any]], rapidapi_key: str, rapidapi_host: str, service_name: str=None) -> APIToolkit:
    """
    Convenience function: create a RapidAPI toolkit
    
    Args:
        schema_path_or_dict: API specification file path or dictionary
        rapidapi_key: RapidAPI key
        rapidapi_host: RapidAPI host
        service_name: Service name (optional)
    
    Returns:
        APIToolkit: Created RapidAPI toolkit
    """
    converter = RapidAPIConverter(input_schema=schema_path_or_dict, description=service_name or '', rapidapi_key=rapidapi_key, rapidapi_host=rapidapi_host)
    return converter.convert_to_toolkit()

def main() -> None:
    """Main function to run condensed converter examples"""
    print('===== API CONVERTER EXAMPLES (CONDENSED) =====')
    rapidapi_test()
    print('\n===== ALL CONDENSED CONVERTER TESTS COMPLETED =====')


# Cluster 10

def _make_catalog_request(search_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Centralized request handler for Catalog API

    Args:
        search_params: Search parameters for the catalog API

    Returns:
        Dict with 'data', 'metadata', and 'error' keys
    """
    try:
        headers = get_auth_headers()
        if 'error' in headers:
            return {'data': [], 'metadata': {}, 'error': headers['error']}
        response = requests.post(CATALOG_API_URL, headers=headers, json=search_params, timeout=TIMEOUT)
        response.raise_for_status()
        raw_data = response.json()
        features = raw_data.get('features', [])
        enhanced_features = []
        for feature in features:
            enhanced_feature = {'id': feature.get('id'), 'collection': feature.get('collection'), 'datetime': feature.get('properties', {}).get('datetime'), 'cloud_cover': feature.get('properties', {}).get('eo:cloud_cover', 0), 'bbox': feature.get('bbox'), 'geometry': feature.get('geometry'), 'properties': feature.get('properties', {}), 'assets': feature.get('assets', {}), 'links': feature.get('links', [])}
            enhanced_features.append(enhanced_feature)
        enhanced_features.sort(key=lambda x: (x.get('datetime', ''), x.get('cloud_cover', 100)), reverse=True)
        return {'data': enhanced_features, 'metadata': {'source': 'Sentinel Hub Catalog API', 'total_scenes': len(enhanced_features), 'search_params': search_params, 'timestamp': datetime.utcnow().isoformat(), 'description': 'Available satellite imagery scenes'}, 'error': None}
    except requests.exceptions.HTTPError as e:
        error_msg = f'Catalog API Error {e.response.status_code}'
        if e.response.status_code == 401:
            error_msg = 'Authentication expired - please check credentials'
        elif e.response.status_code == 403:
            error_msg = 'Insufficient permissions for catalog access'
        elif e.response.status_code == 429:
            error_msg = 'Rate limit exceeded - please try again later'
        return {'data': [], 'metadata': {}, 'error': f'{error_msg}: {str(e)}'}
    except requests.exceptions.Timeout:
        return {'data': [], 'metadata': {}, 'error': 'Catalog API timeout'}
    except requests.exceptions.ConnectionError:
        return {'data': [], 'metadata': {}, 'error': 'Connection error to catalog API'}
    except json.JSONDecodeError:
        return {'data': [], 'metadata': {}, 'error': 'Invalid JSON response from catalog API'}
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Catalog API error: {str(e)}'}

def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers using OAuth2 access token"""
    token_result = get_access_token()
    if token_result.get('error'):
        return {'error': token_result['error']}
    access_token = token_result.get('access_token')
    return {'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json', 'Accept': 'application/json'}

def _make_process_request(process_params: Dict[str, Any], save_to_file: bool=False) -> Dict[str, Any]:
    """
    Centralized request handler for Process API

    Args:
        process_params: Process parameters for the Process API
        save_to_file: Whether to save the image to a temporary file

    Returns:
        Dict with 'data', 'metadata', and 'error' keys
    """
    try:
        headers = get_auth_headers()
        if 'error' in headers:
            return {'data': {}, 'metadata': {}, 'error': headers['error']}
        process_headers = headers.copy()
        process_headers['Accept'] = 'image/*'
        response = requests.post(PROCESS_API_URL, headers=process_headers, json=process_params, timeout=TIMEOUT)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        if save_to_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(response.content)
                file_path = tmp_file.name
            return {'data': {'image_file': file_path, 'content_type': content_type, 'size_bytes': len(response.content)}, 'metadata': {'source': 'Sentinel Hub Process API', 'process_params': process_params, 'timestamp': datetime.utcnow().isoformat(), 'description': 'Processed satellite image saved to file'}, 'error': None}
        else:
            image_b64 = base64.b64encode(response.content).decode('utf-8')
            return {'data': {'image_base64': image_b64, 'content_type': content_type, 'size_bytes': len(response.content)}, 'metadata': {'source': 'Sentinel Hub Process API', 'process_params': process_params, 'timestamp': datetime.utcnow().isoformat(), 'description': 'Processed satellite image (base64 encoded)'}, 'error': None}
    except requests.exceptions.HTTPError as e:
        error_msg = f'Process API Error {e.response.status_code}'
        if e.response.status_code == 401:
            error_msg = 'Authentication expired - please check credentials'
        elif e.response.status_code == 400:
            error_msg = 'Invalid process parameters'
        elif e.response.status_code == 429:
            error_msg = 'Rate limit exceeded - please try again later'
        return {'data': {}, 'metadata': {}, 'error': f'{error_msg}: {str(e)}'}
    except requests.exceptions.Timeout:
        return {'data': {}, 'metadata': {}, 'error': 'Process API timeout - image processing took too long'}
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Process API error: {str(e)}'}

def search_imagery(bbox: List[float], datetime_range: str, collections: Optional[List[str]]=None, max_cloud_cover: float=30.0, limit: int=10) -> Dict[str, Any]:
    """
    Search for available satellite imagery using the Catalog API

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        datetime_range: ISO datetime range "YYYY-MM-DDTHH:MM:SSZ/YYYY-MM-DDTHH:MM:SSZ"
        collections: List of satellite collections to search
        max_cloud_cover: Maximum cloud coverage percentage (default: 30%)
        limit: Maximum number of scenes to return (default: 10)

    Returns:
        Dict with 'data', 'metadata', and 'error' keys containing search results
    """
    try:
        if not bbox or len(bbox) != 4:
            return {'data': [], 'metadata': {}, 'error': 'Bounding box must be a list of 4 coordinates: [min_lon, min_lat, max_lon, max_lat]'}
        if not datetime_range:
            return {'data': [], 'metadata': {}, 'error': 'Date range is required in ISO format'}
        if not collections:
            collections = [SENTINEL_2_L2A]
        search_params = {'bbox': bbox, 'datetime': datetime_range, 'collections': collections, 'limit': limit, 'query': {'eo:cloud_cover': {'lt': max_cloud_cover}}}
        result = _make_catalog_request(search_params)
        if result.get('error'):
            return result
        result['metadata'].update({'bbox': bbox, 'datetime_range': datetime_range, 'collections': collections, 'max_cloud_cover': max_cloud_cover, 'limit': limit})
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error searching imagery: {str(e)}'}

def search_imagery_by_coordinates(lat: float, lon: float, radius_km: float=10.0, start_date: str=None, end_date: str=None, collections: Optional[List[str]]=None, max_cloud_cover: float=30.0, limit: int=10) -> Dict[str, Any]:
    """
    Search for satellite imagery by center coordinates and radius

    Args:
        lat: Latitude of center point
        lon: Longitude of center point
        radius_km: Search radius in kilometers (default: 10km)
        start_date: Start date in YYYY-MM-DD format (default: 30 days ago)
        end_date: End date in YYYY-MM-DD format (default: today)
        collections: List of satellite collections to search
        max_cloud_cover: Maximum cloud coverage percentage (default: 30%)
        limit: Maximum number of scenes to return (default: 10)

    Returns:
        Dict with 'data', 'metadata', and 'error' keys containing search results
    """
    try:
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * abs(lat) if lat != 0 else 111.0)
        bbox = [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]
        if not end_date:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        datetime_range = f'{start_date}T00:00:00Z/{end_date}T23:59:59Z'
        result = search_imagery(bbox, datetime_range, collections, max_cloud_cover, limit)
        if result.get('error'):
            return result
        result['metadata'].update({'search_center': {'lat': lat, 'lon': lon}, 'search_radius_km': radius_km, 'search_type': 'coordinate_based'})
        return result
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Error searching by coordinates: {str(e)}'}

def process_imagery(bbox: List[float], datetime_range: str, evalscript: str=None, evalscript_type: str='true_color', width: int=512, height: int=512, format_type: str='image/png', save_to_file: bool=False) -> Dict[str, Any]:
    """
    Process satellite imagery using the Process API

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        datetime_range: ISO datetime range
        evalscript: Custom evalscript (overrides evalscript_type)
        evalscript_type: Predefined evalscript type (true_color, false_color, ndvi, etc.)
        width: Output image width (default: 512)
        height: Output image height (default: 512)
        format_type: Output format (image/png, image/tiff, etc.)
        save_to_file: Whether to save image to temporary file (default: False)

    Returns:
        Dict with 'data', 'metadata', and 'error' keys containing processed image
    """
    try:
        if not bbox or len(bbox) != 4:
            return {'data': {}, 'metadata': {}, 'error': 'Bounding box must be a list of 4 coordinates'}
        if not datetime_range:
            return {'data': {}, 'metadata': {}, 'error': 'Date range is required'}
        if not evalscript and evalscript_type in EVALSCRIPTS:
            evalscript = EVALSCRIPTS[evalscript_type]
        elif not evalscript:
            evalscript = EVALSCRIPTS['true_color']
        process_params = {'input': {'bounds': {'bbox': bbox}, 'data': [{'type': SENTINEL_2_L2A, 'dataFilter': {'timeRange': {'from': datetime_range.split('/')[0], 'to': datetime_range.split('/')[1]}, 'maxCloudCoverage': max(0, min(100, 30))}}]}, 'output': {'width': width, 'height': height, 'responses': [{'identifier': 'default', 'format': {'type': format_type}}]}, 'evalscript': evalscript}
        result = _make_process_request(process_params, save_to_file)
        if result.get('error'):
            return result
        result['metadata'].update({'bbox': bbox, 'datetime_range': datetime_range, 'evalscript_type': evalscript_type, 'width': width, 'height': height, 'format': format_type})
        return result
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Error processing imagery: {str(e)}'}

def process_imagery_by_scene_id(scene_id: str, evalscript: str=None, evalscript_type: str='true_color', width: int=512, height: int=512, format_type: str='image/png', save_to_file: bool=False) -> Dict[str, Any]:
    """
    Process satellite imagery using a specific scene ID

    Args:
        scene_id: Scene ID from catalog search
        evalscript: Custom evalscript (overrides evalscript_type)
        evalscript_type: Predefined evalscript type
        width: Output image width
        height: Output image height
        format_type: Output format
        save_to_file: Whether to save image to temporary file

    Returns:
        Dict with 'data', 'metadata', and 'error' keys containing processed image
    """
    try:
        if not scene_id:
            return {'data': {}, 'metadata': {}, 'error': 'Scene ID is required'}
        scene_datetime = None
        bbox = None
        import re
        date_match = re.search('_(\\d{8}T\\d{6})_', scene_id)
        if date_match:
            scene_datetime = date_match.group(1)
        if not scene_datetime:
            return {'data': {}, 'metadata': {}, 'error': 'Could not extract datetime from scene ID'}
        scene_time = datetime.strptime(scene_datetime, '%Y%m%dT%H%M%S')
        datetime_range = f'{scene_time.strftime('%Y-%m-%dT%H:%M:%SZ')}/{scene_time.strftime('%Y-%m-%dT%H:%M:%SZ')}'
        search_result = search_imagery(bbox=[-180, -90, 180, 90], datetime_range=datetime_range, limit=1)
        if search_result.get('error') or not search_result.get('data'):
            return {'data': {}, 'metadata': {}, 'error': f'Could not find scene with ID: {scene_id}'}
        scene_data = search_result['data'][0]
        bbox = scene_data.get('bbox')
        if not bbox:
            return {'data': {}, 'metadata': {}, 'error': 'Could not determine bounding box for scene'}
        result = process_imagery(bbox=bbox, datetime_range=datetime_range, evalscript=evalscript, evalscript_type=evalscript_type, width=width, height=height, format_type=format_type, save_to_file=save_to_file)
        if result.get('error'):
            return result
        result['metadata'].update({'scene_id': scene_id, 'processing_method': 'scene_id_based'})
        return result
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Error processing scene {scene_id}: {str(e)}'}

def test_api_connectivity() -> Dict[str, Any]:
    """
    Test connectivity to Sentinel Hub APIs

    Returns:
        Dict with connectivity test results
    """
    results = {}
    try:
        auth_result = get_access_token()
        results['authentication'] = {'status': 'success' if not auth_result.get('error') else 'error', 'message': auth_result.get('error') or 'Authentication successful', 'token_expires_in': auth_result.get('expires_in')}
    except Exception as e:
        results['authentication'] = {'status': 'error', 'message': str(e)}
    try:
        test_bbox = [13.0, 45.0, 13.1, 45.1]
        test_datetime = f'{(datetime.utcnow() - timedelta(days=60)).strftime('%Y-%m-%d')}T00:00:00Z/{datetime.utcnow().strftime('%Y-%m-%d')}T23:59:59Z'
        catalog_result = search_imagery(bbox=test_bbox, datetime_range=test_datetime, limit=1)
        results['catalog_api'] = {'status': 'success' if not catalog_result.get('error') else 'error', 'message': catalog_result.get('error') or 'Catalog API working', 'scenes_found': len(catalog_result.get('data', []))}
    except Exception as e:
        results['catalog_api'] = {'status': 'error', 'message': str(e)}
    return {'data': results, 'metadata': {'test_timestamp': datetime.utcnow().isoformat(), 'base_url': BASE_URL, 'credentials_configured': bool(os.environ.get('SENTINELHUB_CLIENT_ID') and os.environ.get('SENTINELHUB_CLIENT_SECRET'))}, 'error': None}

def get_access_token() -> Dict[str, Any]:
    """
    Get OAuth2 access token using client credentials

    Returns:
        Dict with access token or error information
    """
    try:
        client_id = os.environ.get('SENTINELHUB_CLIENT_ID')
        client_secret = os.environ.get('SENTINELHUB_CLIENT_SECRET')
        if not client_id or not client_secret:
            return {'error': 'Missing Sentinel Hub credentials. Set SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET environment variables.'}
        auth_string = f'{client_id}:{client_secret}'
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        headers = {'Authorization': f'Basic {auth_b64}', 'Content-Type': 'application/x-www-form-urlencoded'}
        data = 'grant_type=client_credentials'
        response = requests.post(TOKEN_URL, headers=headers, data=data, timeout=TIMEOUT)
        response.raise_for_status()
        token_data = response.json()
        return {'access_token': token_data.get('access_token'), 'expires_in': token_data.get('expires_in'), 'token_type': token_data.get('token_type'), 'error': None}
    except requests.exceptions.HTTPError as e:
        error_msg = f'Authentication failed: {e.response.status_code}'
        if e.response.status_code == 401:
            error_msg = 'Invalid client credentials'
        elif e.response.status_code == 403:
            error_msg = 'Access forbidden'
        return {'error': f'{error_msg}: {str(e)}'}
    except requests.exceptions.Timeout:
        return {'error': 'Authentication timeout'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Connection error during authentication'}
    except Exception as e:
        return {'error': f'Authentication error: {str(e)}'}

def main():
    """Command-line interface for Sentinel Hub API wrapper"""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: python sentinelhub_data.py <command> [args]', 'available_commands': ['search <bbox> <datetime_range> [collections] [max_cloud] [limit]', 'search-coords <lat> <lon> <radius_km> [start_date] [end_date] [collections]', 'process <bbox> <datetime_range> [evalscript_type] [width] [height] [format] [save_to_file]', 'process-scene <scene_id> [evalscript_type] [width] [height] [format] [save_to_file]', 'collections', 'evalscripts', 'test-connectivity'], 'examples': ['sentinelhub_data.py search "13.0,45.0,14.0,46.0" "2019-12-10T00:00:00Z/2019-12-10T23:59:59Z" sentinel-2-l2a 20 5', 'sentinelhub_data.py search-coords 45.5 13.6 10 2019-12-01 2019-12-31', 'sentinelhub_data.py process "13.0,45.0,14.0,46.0" "2019-12-10T00:00:00Z/2019-12-10T23:59:59Z" ndvi 1024 1024', 'sentinelhub_data.py process-scene S2A_MSIL2A_20191210T100311_N0213_R122_T33TUE_20191210T121921 true_color', 'sentinelhub_data.py collections', 'sentinelhub_data.py evalscripts', 'sentinelhub_data.py test-connectivity']}, indent=2))
        sys.exit(1)
    command = sys.argv[1]
    try:
        if command == 'search':
            if len(sys.argv) < 4:
                result = {'error': 'Usage: search <bbox> <datetime_range> [collections] [max_cloud] [limit]'}
            else:
                bbox = json.loads(sys.argv[2])
                datetime_range = sys.argv[3]
                collections = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
                max_cloud = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
                limit = int(sys.argv[6]) if len(sys.argv) > 6 else 10
                result = search_imagery(bbox, datetime_range, collections, max_cloud, limit)
        elif command == 'search-coords':
            if len(sys.argv) < 4:
                result = {'error': 'Usage: search-coords <lat> <lon> <radius_km> [start_date] [end_date] [collections]'}
            else:
                lat = float(sys.argv[2])
                lon = float(sys.argv[3])
                radius = float(sys.argv[4])
                start_date = sys.argv[5] if len(sys.argv) > 5 else None
                end_date = sys.argv[6] if len(sys.argv) > 6 else None
                collections = json.loads(sys.argv[7]) if len(sys.argv) > 7 else None
                result = search_imagery_by_coordinates(lat, lon, radius, start_date, end_date, collections)
        elif command == 'process':
            if len(sys.argv) < 4:
                result = {'error': 'Usage: process <bbox> <datetime_range> [evalscript_type] [width] [height] [format] [save_to_file]'}
            else:
                bbox = json.loads(sys.argv[2])
                datetime_range = sys.argv[3]
                evalscript_type = sys.argv[4] if len(sys.argv) > 4 else 'true_color'
                width = int(sys.argv[5]) if len(sys.argv) > 5 else 512
                height = int(sys.argv[6]) if len(sys.argv) > 6 else 512
                format_type = sys.argv[7] if len(sys.argv) > 7 else 'image/png'
                save_to_file = sys.argv[8].lower() == 'true' if len(sys.argv) > 8 else False
                result = process_imagery(bbox, datetime_range, None, evalscript_type, width, height, format_type, save_to_file)
        elif command == 'process-scene':
            if len(sys.argv) < 3:
                result = {'error': 'Usage: process-scene <scene_id> [evalscript_type] [width] [height] [format] [save_to_file]'}
            else:
                scene_id = sys.argv[2]
                evalscript_type = sys.argv[3] if len(sys.argv) > 3 else 'true_color'
                width = int(sys.argv[4]) if len(sys.argv) > 4 else 512
                height = int(sys.argv[5]) if len(sys.argv) > 5 else 512
                format_type = sys.argv[6] if len(sys.argv) > 6 else 'image/png'
                save_to_file = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False
                result = process_imagery_by_scene_id(scene_id, None, evalscript_type, width, height, format_type, save_to_file)
        elif command == 'collections':
            result = get_available_collections()
        elif command == 'evalscripts':
            result = get_evalscript_types()
        elif command == 'test-connectivity':
            result = test_api_connectivity()
        else:
            result = {'error': f'Unknown command: {command}', 'available_commands': ['search <bbox> <datetime_range> [collections] [max_cloud] [limit]', 'search-coords <lat> <lon> <radius_km> [start_date] [end_date] [collections]', 'process <bbox> <datetime_range> [evalscript_type] [width] [height] [format] [save_to_file]', 'process-scene <scene_id> [evalscript_type] [width] [height] [format] [save_to_file]', 'collections', 'evalscripts', 'test-connectivity']}
        print(json.dumps(result, indent=2))
    except json.JSONDecodeError as e:
        print(json.dumps({'error': f'Invalid JSON parameter: {str(e)}'}))
        sys.exit(1)
    except ValueError as e:
        print(json.dumps({'error': f'Invalid parameter: {str(e)}'}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': f'Command execution failed: {str(e)}'}))
        sys.exit(1)

def get_available_collections() -> Dict[str, Any]:
    """
    Get list of available satellite collections with descriptions

    Returns:
        Dict with collection information
    """
    return {'data': [{'id': SENTINEL_2_L2A, 'name': 'Sentinel-2 Level-2A', 'description': 'High-resolution optical imagery with atmospheric correction', 'resolution': '10m, 20m, 60m', 'revisit_time': '5 days', 'bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'], 'use_cases': ['Vegetation monitoring', 'Land cover', 'Coastal areas']}, {'id': SENTINEL_1_GRD, 'name': 'Sentinel-1 Ground Range Detected', 'description': 'Radar imagery (C-band) - works day and night, through clouds', 'resolution': '5x20m', 'revisit_time': '1-3 days', 'bands': ['VV', 'VH', 'HH', 'HV'], 'use_cases': ['Flood monitoring', 'Oil spill detection', 'Ship detection']}, {'id': LANDSAT_8_L1, 'name': 'Landsat 8 Level-1', 'description': 'Medium-resolution optical imagery', 'resolution': '15m, 30m, 100m', 'revisit_time': '16 days', 'bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11'], 'use_cases': ['Land cover change', 'Urban development', 'Agriculture']}, {'id': MODIS, 'name': 'MODIS', 'description': 'Daily global coverage for environmental monitoring', 'resolution': '250m, 500m, 1000m', 'revisit_time': 'Daily', 'bands': ['Multiple spectral bands'], 'use_cases': ['Climate monitoring', 'Vegetation indices', 'Disaster monitoring']}], 'metadata': {'source': 'Sentinel Hub', 'timestamp': datetime.utcnow().isoformat(), 'description': 'Available satellite collections'}, 'error': None}

def get_evalscript_types() -> Dict[str, Any]:
    """
    Get list of available evalscript types with descriptions

    Returns:
        Dict with evalscript information
    """
    return {'data': [{'id': 'true_color', 'name': 'True Color', 'description': 'Natural color image as seen by human eye', 'bands': ['B04 (Red)', 'B03 (Green)', 'B02 (Blue)'], 'use_cases': ['General visualization', 'Human geography']}, {'id': 'false_color', 'name': 'False Color (Infrared)', 'description': 'Infrared composite highlighting vegetation', 'bands': ['B08 (NIR)', 'B04 (Red)', 'B03 (Green)'], 'use_cases': ['Vegetation health', 'Forest monitoring']}, {'id': 'ndvi', 'name': 'NDVI (Normalized Difference Vegetation Index)', 'description': 'Vegetation health indicator', 'formula': '(NIR - Red) / (NIR + Red)', 'range': '-1 to 1', 'use_cases': ['Crop monitoring', 'Drought assessment', 'Yield prediction']}, {'id': 'ndwi', 'name': 'NDWI (Normalized Difference Water Index)', 'description': 'Water body detection and monitoring', 'formula': '(Green - NIR) / (Green + NIR)', 'range': '-1 to 1', 'use_cases': ['Flood monitoring', 'Water resource management', 'Coastal monitoring']}, {'id': 'urban_index', 'name': 'Urban Index', 'description': 'Built-up area detection', 'formula': '(SWIR - NIR) / (SWIR + NIR)', 'range': '-1 to 1', 'use_cases': ['Urban sprawl monitoring', 'Construction tracking', 'Infrastructure planning']}], 'metadata': {'source': 'Sentinel Hub', 'timestamp': datetime.utcnow().isoformat(), 'description': 'Available image processing types'}, 'error': None}

def get_collections(page: int=1) -> Dict[str, Any]:
    """
    Get all available collections (catalogues) from data.gov.sg.

    Args:
        page: Page number for pagination (default: 1)

    Returns:
        Dict with 'data', 'metadata', and 'error' keys
    """
    try:
        url = f'{COLLECTION_API_BASE}/collections'
        params = {'page': page}
        headers = get_headers()
        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        raw_data = response.json()
        enhanced_data = []
        if isinstance(raw_data, dict) and 'data' in raw_data and ('collections' in raw_data['data']):
            for collection in raw_data['data']['collections']:
                enhanced_collection = {'id': collection.get('collectionId'), 'name': collection.get('name'), 'description': collection.get('description'), 'frequency': collection.get('frequency'), 'sources': collection.get('sources', []), 'managed_by': collection.get('managedByAgencyName'), 'created_at': collection.get('createdAt'), 'updated_at': collection.get('lastUpdatedAt')}
                enhanced_data.append(enhanced_collection)
        metadata = {'page': page, 'total_count': len(enhanced_data), 'api_used': 'collections', 'timestamp': datetime.utcnow().isoformat()}
        return {'data': enhanced_data, 'metadata': metadata, 'error': None}
    except requests.exceptions.HTTPError as e:
        return {'data': [], 'metadata': {}, 'error': f'HTTP error fetching collections: {str(e)}'}
    except requests.exceptions.ConnectionError:
        return {'data': [], 'metadata': {}, 'error': 'Connection error: Could not connect to data.gov.sg API'}
    except requests.exceptions.Timeout:
        return {'data': [], 'metadata': {}, 'error': 'Request timeout: Could not fetch collections within time limit'}
    except json.JSONDecodeError:
        return {'data': [], 'metadata': {}, 'error': 'Invalid JSON response from collections API'}
    except Exception as e:
        return {'data': [], 'metadata': {}, 'error': f'Unexpected error fetching collections: {str(e)}'}

def get_headers() -> Dict[str, str]:
    """Get standard headers for API requests."""
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    if API_KEY:
        headers['x-api-key'] = API_KEY
    return headers

def get_collection_details(collection_id: Union[str, int], with_dataset_metadata: bool=False) -> Dict[str, Any]:
    """
    Get detailed information about a specific collection including its datasets.

    Args:
        collection_id: The ID of the collection
        with_dataset_metadata: Include full metadata for datasets (default: False)

    Returns:
        Dict with 'data', 'metadata', and 'error' keys
    """
    try:
        url = f'{COLLECTION_API_BASE}/collections/{collection_id}'
        params = {'withDatasetMetadata': str(with_dataset_metadata).lower()}
        headers = get_headers()
        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        raw_data = response.json()
        collection_data = {}
        datasets_data = []
        if isinstance(raw_data, dict) and 'data' in raw_data:
            collection_info = raw_data['data']
            collection_data = {'id': collection_info.get('collectionId'), 'name': collection_info.get('name'), 'description': collection_info.get('description'), 'frequency': collection_info.get('frequency'), 'sources': collection_info.get('sources', []), 'managed_by': collection_info.get('managedByAgencyName'), 'created_at': collection_info.get('createdAt'), 'updated_at': collection_info.get('lastUpdatedAt')}
            if 'datasets' in collection_info:
                for dataset in collection_info['datasets']:
                    enhanced_dataset = {'id': dataset.get('datasetId'), 'name': dataset.get('name'), 'description': dataset.get('description'), 'created_at': dataset.get('createdAt'), 'updated_at': dataset.get('lastUpdatedAt'), 'metadata': dataset.get('metadata', {})}
                    datasets_data.append(enhanced_dataset)
        result_data = {'collection': collection_data, 'datasets': datasets_data}
        metadata = {'collection_id': collection_id, 'dataset_count': len(datasets_data), 'with_dataset_metadata': with_dataset_metadata, 'api_used': 'collection_details', 'timestamp': datetime.utcnow().isoformat()}
        return {'data': result_data, 'metadata': metadata, 'error': None}
    except requests.exceptions.HTTPError as e:
        error_msg = f'HTTP error fetching collection details: {str(e)}'
        if e.response.status_code == 404:
            error_msg = f'Collection with ID {collection_id} not found'
        return {'data': {}, 'metadata': {}, 'error': error_msg}
    except requests.exceptions.ConnectionError:
        return {'data': {}, 'metadata': {}, 'error': 'Connection error: Could not connect to data.gov.sg API'}
    except requests.exceptions.Timeout:
        return {'data': {}, 'metadata': {}, 'error': 'Request timeout: Could not fetch collection details within time limit'}
    except json.JSONDecodeError:
        return {'data': {}, 'metadata': {}, 'error': 'Invalid JSON response from collection details API'}
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Unexpected error fetching collection details: {str(e)}'}

def initiate_dataset_download(dataset_id: str, column_names: Optional[List[str]]=None, filters: Optional[List[Dict[str, Any]]]=None) -> Dict[str, Any]:
    """
    Initiate a download for a specific dataset with optional filtering.

    Args:
        dataset_id: The ID of the dataset
        column_names: Optional list of column names to include
        filters: Optional list of filter objects

    Returns:
        Dict with 'data', 'metadata', and 'error' keys containing download initiation info
    """
    try:
        url = f'{DATASET_API_BASE}/{dataset_id}/initiate-download'
        headers = get_headers()
        request_body = {}
        if column_names:
            request_body['columnNames'] = column_names
        if filters:
            request_body['filters'] = {'filters': filters}
        response = requests.get(url, json=request_body if request_body else None, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        raw_data = response.json()
        enhanced_data = {'dataset_id': dataset_id, 'request_id': raw_data.get('requestId'), 'status': raw_data.get('status', 'INITIATED'), 'message': raw_data.get('message'), 'timestamp': datetime.utcnow().isoformat()}
        metadata = {'dataset_id': dataset_id, 'column_names': column_names, 'filters_count': len(filters) if filters else 0, 'api_used': 'initiate_download', 'timestamp': datetime.utcnow().isoformat()}
        return {'data': enhanced_data, 'metadata': metadata, 'error': None}
    except requests.exceptions.HTTPError as e:
        error_msg = f'HTTP error initiating dataset download: {str(e)}'
        if e.response.status_code == 404:
            error_msg = f'Dataset with ID {dataset_id} not found'
        return {'data': {}, 'metadata': {}, 'error': error_msg}
    except requests.exceptions.ConnectionError:
        return {'data': {}, 'metadata': {}, 'error': 'Connection error: Could not connect to data.gov.sg API'}
    except requests.exceptions.Timeout:
        return {'data': {}, 'metadata': {}, 'error': 'Request timeout: Could not initiate dataset download within time limit'}
    except json.JSONDecodeError:
        return {'data': {}, 'metadata': {}, 'error': 'Invalid JSON response from dataset download API'}
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Unexpected error initiating dataset download: {str(e)}'}

def poll_dataset_download(dataset_id: str) -> Dict[str, Any]:
    """
    Poll the status of a dataset download and get download URL when ready.

    Args:
        dataset_id: The ID of the dataset

    Returns:
        Dict with 'data', 'metadata', and 'error' keys containing download status and URL
    """
    try:
        url = f'{DATASET_API_BASE}/{dataset_id}/poll-download'
        headers = get_headers()
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        raw_data = response.json()
        enhanced_data = {'dataset_id': dataset_id, 'status': raw_data.get('data', {}).get('status'), 'download_url': raw_data.get('data', {}).get('url'), 'error_message': raw_data.get('errorMsg'), 'code': raw_data.get('code'), 'timestamp': datetime.utcnow().isoformat()}
        metadata = {'dataset_id': dataset_id, 'api_used': 'poll_download', 'timestamp': datetime.utcnow().isoformat()}
        return {'data': enhanced_data, 'metadata': metadata, 'error': None}
    except requests.exceptions.HTTPError as e:
        return {'data': {}, 'metadata': {}, 'error': f'HTTP error polling dataset download: {str(e)}'}
    except requests.exceptions.ConnectionError:
        return {'data': {}, 'metadata': {}, 'error': 'Connection error: Could not connect to data.gov.sg API'}
    except requests.exceptions.Timeout:
        return {'data': {}, 'metadata': {}, 'error': 'Request timeout: Could not poll dataset download within time limit'}
    except json.JSONDecodeError:
        return {'data': {}, 'metadata': {}, 'error': 'Invalid JSON response from dataset poll API'}
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Unexpected error polling dataset download: {str(e)}'}

def get_pm25_readings(date: Optional[str]=None, pagination_token: Optional[str]=None) -> Dict[str, Any]:
    """
    Get PM2.5 readings from Singapore's air quality monitoring stations.

    Args:
        date: Optional date filter (YYYY-MM-DD or YYYY-MM-DDTHH:mm:ss)
        pagination_token: Optional token for pagination

    Returns:
        Dict with 'data', 'metadata', and 'error' keys
    """
    try:
        url = f'{REALTIME_API_BASE}/pm25'
        params = {}
        if date:
            params['date'] = date
        if pagination_token:
            params['paginationToken'] = pagination_token
        headers = get_headers()
        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        raw_data = response.json()
        enhanced_data = {'region_metadata': [], 'readings': [], 'timestamp': datetime.utcnow().isoformat()}
        if isinstance(raw_data, dict) and 'data' in raw_data:
            data = raw_data['data']
            if 'regionMetadata' in data:
                for region in data['regionMetadata']:
                    enhanced_region = {'name': region.get('name'), 'label_location': region.get('labelLocation'), 'latitude': region.get('labelLocation', {}).get('latitude') if isinstance(region.get('labelLocation'), dict) else None, 'longitude': region.get('labelLocation', {}).get('longitude') if isinstance(region.get('labelLocation'), dict) else None}
                    enhanced_data['region_metadata'].append(enhanced_region)
            if 'items' in data and data['items']:
                for item in data['items']:
                    reading_data = {'timestamp': item.get('timestamp'), 'date': item.get('date'), 'readings': item.get('readings', {}), 'update_timestamp': item.get('updatedTimestamp')}
                    enhanced_data['readings'].append(reading_data)
        metadata = {'api_used': 'pm25_realtime', 'date_filter': date, 'has_pagination_token': bool(pagination_token), 'region_count': len(enhanced_data['region_metadata']), 'reading_count': len(enhanced_data['readings']), 'timestamp': datetime.utcnow().isoformat()}
        return {'data': enhanced_data, 'metadata': metadata, 'error': None}
    except requests.exceptions.HTTPError as e:
        return {'data': {}, 'metadata': {}, 'error': f'HTTP error fetching PM2.5 readings: {str(e)}'}
    except requests.exceptions.ConnectionError:
        return {'data': {}, 'metadata': {}, 'error': 'Connection error: Could not connect to data.gov.sg API'}
    except requests.exceptions.Timeout:
        return {'data': {}, 'metadata': {}, 'error': 'Request timeout: Could not fetch PM2.5 readings within time limit'}
    except json.JSONDecodeError:
        return {'data': {}, 'metadata': {}, 'error': 'Invalid JSON response from PM2.5 API'}
    except Exception as e:
        return {'data': {}, 'metadata': {}, 'error': f'Unexpected error fetching PM2.5 readings: {str(e)}'}

def test_api_connectivity() -> Dict[str, Any]:
    """
    Test basic connectivity to all API endpoints.

    Returns:
        Dict with connectivity test results for each endpoint
    """
    results = {}
    try:
        collections_response = get_collections(page=1)
        results['collections_api'] = {'status': 'connected' if not collections_response['error'] else 'error', 'message': collections_response['error'] or 'Successfully connected', 'response_time_ms': 0}
    except Exception as e:
        results['collections_api'] = {'status': 'error', 'message': str(e), 'response_time_ms': 0}
    try:
        pm25_response = get_pm25_readings()
        results['realtime_api'] = {'status': 'connected' if not pm25_response['error'] else 'error', 'message': pm25_response['error'] or 'Successfully connected', 'response_time_ms': 0}
    except Exception as e:
        results['realtime_api'] = {'status': 'error', 'message': str(e), 'response_time_ms': 0}
    return {'data': results, 'metadata': {'test_timestamp': datetime.utcnow().isoformat(), 'api_key_configured': bool(API_KEY)}, 'error': None}

def main():
    """Command-line interface for data.gov.sg API wrapper."""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: python datagovsg_data.py <command> [args]', 'available_commands': ['collections [page]', 'collection-details <collection_id> [with_metadata]', 'initiate-download <dataset_id>', 'poll-download <dataset_id>', 'pm25 [date]', 'test-connectivity']}))
        sys.exit(1)
    command = sys.argv[1]
    try:
        if command == 'collections':
            page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            result = get_collections(page=page)
        elif command == 'collection-details':
            if len(sys.argv) < 3:
                print(json.dumps({'error': 'Usage: datagovsg_data.py collection-details <collection_id> [with_metadata]'}))
                sys.exit(1)
            collection_id = sys.argv[2]
            with_metadata = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
            result = get_collection_details(collection_id, with_dataset_metadata=with_metadata)
        elif command == 'initiate-download':
            if len(sys.argv) < 3:
                print(json.dumps({'error': 'Usage: datagovsg_data.py initiate-download <dataset_id>'}))
                sys.exit(1)
            dataset_id = sys.argv[2]
            result = initiate_dataset_download(dataset_id)
        elif command == 'poll-download':
            if len(sys.argv) < 3:
                print(json.dumps({'error': 'Usage: datagovsg_data.py poll-download <dataset_id>'}))
                sys.exit(1)
            dataset_id = sys.argv[2]
            result = poll_dataset_download(dataset_id)
        elif command == 'pm25':
            date = sys.argv[2] if len(sys.argv) > 2 else None
            result = get_pm25_readings(date=date)
        elif command == 'test-connectivity':
            result = test_api_connectivity()
        else:
            result = {'error': f'Unknown command: {command}', 'available_commands': ['collections [page]', 'collection-details <collection_id> [with_metadata]', 'initiate-download <dataset_id>', 'poll-download <dataset_id>', 'pm25 [date]', 'test-connectivity']}
        print(json.dumps(result, indent=2))
    except ValueError as e:
        print(json.dumps({'error': f'Invalid parameter: {str(e)}'}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({'error': f'Command execution failed: {str(e)}'}))
        sys.exit(1)


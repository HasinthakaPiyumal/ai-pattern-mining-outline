# Cluster 64

def fetch_collections():
    """Fetch available collections and their datasets"""
    global datasets
    dpg.set_value('status_text', 'Loading all datasets... This may take a moment')

    def fetch_thread():
        global datasets
        datasets_list = []
        print('DEBUG: Scanning collections 1-100...')
        for collection_id in range(1, 101):
            try:
                url = f'https://api-production.data.gov.sg/v2/public/api/collections/{collection_id}/metadata'
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    collection_meta = data.get('data', {}).get('collectionMetadata', {})
                    collection_name = collection_meta.get('name', f'Collection {collection_id}')
                    child_datasets = collection_meta.get('childDatasets', [])
                    print(f'DEBUG: Collection {collection_id} ({collection_name}): {len(child_datasets)} datasets')
                    for idx, dataset_id in enumerate(child_datasets):
                        datasets_list.append({'display_name': f'[C{collection_id}] {collection_name} - Dataset {idx + 1}', 'dataset_id': dataset_id, 'collection_id': collection_id})
                else:
                    print(f'DEBUG: Collection {collection_id}: {response.status_code}')
            except Exception as e:
                print(f'DEBUG: Collection {collection_id}: Error - {e}')
                continue
            if collection_id % 10 == 0:
                dpg.set_value('status_text', f'Scanned {collection_id} collections... Found {len(datasets_list)} datasets')
        print('DEBUG: Attempting direct dataset discovery...')
        try:
            api_endpoints = ['https://api-production.data.gov.sg/v2/public/api/datasets', 'https://api.data.gov.sg/v1/public/api/datasets', 'https://data.gov.sg/api/action/package_list']
            for endpoint in api_endpoints:
                try:
                    print(f'DEBUG: Trying endpoint: {endpoint}')
                    response = requests.get(endpoint, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        print(f'DEBUG: Endpoint response keys: {(list(data.keys()) if isinstance(data, dict) else 'Not a dict')}')
                        if 'result' in data and isinstance(data['result'], list):
                            for dataset_name in data['result'][:100]:
                                datasets_list.append({'display_name': f'[DIRECT] {dataset_name}', 'dataset_id': dataset_name, 'collection_id': 'direct'})
                        elif 'data' in data:
                            dataset_data = data['data']
                            if isinstance(dataset_data, list):
                                for dataset in dataset_data[:100]:
                                    if isinstance(dataset, dict) and 'id' in dataset:
                                        datasets_list.append({'display_name': f'[DIRECT] {dataset.get('name', dataset['id'])}', 'dataset_id': dataset['id'], 'collection_id': 'direct'})
                except Exception as e:
                    print(f'DEBUG: Endpoint {endpoint} failed: {e}')
                    continue
        except Exception as e:
            print(f'DEBUG: Direct discovery failed: {e}')
        print('DEBUG: Attempting systematic dataset ID discovery...')
        id_patterns = ['d_{}', 'dataset_{}', 'sg_{}']
        known_working_samples = ['3f960c10fed6145404ca7b821f263b87', 'af2042c77ffaf0db5d75561ce9ef5688']
        for sample_id in known_working_samples:
            for i in range(50):
                try:
                    test_id = sample_id[:-2] + f'{i:02d}'
                    test_url = f'https://data.gov.sg/api/action/datastore_search?resource_id=d_{test_id}&limit=1'
                    response = requests.get(test_url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('success'):
                            datasets_list.append({'display_name': f'[DISCOVERED] Dataset d_{test_id}', 'dataset_id': f'd_{test_id}', 'collection_id': 'discovered'})
                            print(f'DEBUG: Found working dataset: d_{test_id}')
                except:
                    continue
        print(f'DEBUG: Total datasets found: {len(datasets_list)}')
        datasets = datasets_list
        dataset_names = [d['display_name'] for d in datasets]
        dpg.configure_item('dataset_combo', items=dataset_names)
        dpg.set_value('dataset_combo', 'Select a dataset...')
        dpg.set_value('status_text', f'Loaded {len(datasets)} datasets from {max([d['collection_id'] for d in datasets if str(d['collection_id']).isdigit()] + [0])} collections')
    threading.Thread(target=fetch_thread, daemon=True).start()


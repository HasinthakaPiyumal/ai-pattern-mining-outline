# Cluster 63

def on_prev_page():
    """Go to previous page"""
    global current_page, records_per_page
    if current_page > 1:
        offset = (current_page - 2) * records_per_page
        fetch_dataset_data(offset)

def fetch_dataset_data(offset=0):
    """Fetch data for selected dataset with pagination"""
    global current_data, records_per_page
    if not selected_dataset:
        dpg.set_value('status_text', 'Please select a dataset first')
        return
    dpg.set_value('status_text', 'Fetching data...')

    def fetch_thread():
        global current_data
        try:
            dataset_id = selected_dataset['dataset_id']
            print(f'DEBUG: Fetching data for dataset: {dataset_id}')
            url = f'https://data.gov.sg/api/action/datastore_search?resource_id={dataset_id}'
            url += f'&limit={records_per_page}&offset={offset}'
            print(f'DEBUG: API URL: {url}')
            response = requests.get(url, timeout=15)
            print(f'DEBUG: Response status: {response.status_code}')
            if response.status_code == 200:
                data = response.json()
                print(f'DEBUG: Response success: {data.get('success')}')
                if data.get('success'):
                    result = data.get('result', {})
                    records = result.get('records', [])
                    total = result.get('total', 0)
                    print(f'DEBUG: Found {len(records)} records out of {total} total')
                    current_data = {'records': records, 'total': total, 'offset': offset, 'fields': result.get('fields', [])}
                    update_data_display()
                    update_pagination_info()
                    dpg.set_value('status_text', f'Showing {len(records)} of {total} records')
                else:
                    error_msg = data.get('error', {})
                    print(f'DEBUG: API Error: {error_msg}')
                    dpg.set_value('status_text', f'API Error: {error_msg.get('message', 'Unknown error')}')
            else:
                print(f'DEBUG: HTTP Error: {response.status_code}')
                dpg.set_value('status_text', f'HTTP Error: {response.status_code}')
        except Exception as e:
            print(f'DEBUG: Exception: {str(e)}')
            dpg.set_value('status_text', f'Error: {str(e)}')
    threading.Thread(target=fetch_thread, daemon=True).start()

def on_next_page():
    """Go to next page"""
    global current_page, records_per_page, current_data
    total = current_data.get('total', 0)
    max_pages = (total + records_per_page - 1) // records_per_page
    if current_page < max_pages:
        offset = current_page * records_per_page
        fetch_dataset_data(offset)


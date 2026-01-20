# Cluster 21

def download_file_from_google_drive(file_name, destination):
    """
    Downloads a zip file from google drive. Assumes the file has open access. 
    Args:
        file_name: name of the file to download
        destination: path to save the file to
    """
    URL = 'https://drive.google.com/uc?export=download'
    id = file_id_map[file_name]
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


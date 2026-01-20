# Cluster 2

def list_s3_files(s3_location):
    if s3_location[-1] != '/':
        s3_location = s3_location + '/'
    file_path_list = []
    xml = make_get_request(s3_location)
    s3_listing = xmltodict.parse(xml)
    s3_content_metadata = s3_listing['ListBucketResult']['Contents']
    for index, metadata in enumerate(s3_content_metadata):
        file_path = s3_content_metadata[index]['Key']
        file_path_list.append(s3_location + file_path)
    return file_path_list

def make_get_request(url):
    response = requests.get(url)
    return response.content.decode('utf-8')


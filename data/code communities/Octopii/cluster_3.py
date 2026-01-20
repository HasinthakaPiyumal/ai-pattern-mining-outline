# Cluster 3

def print_logo():
    logo = '⠀⠀⠀ ⠀⡀⠀⠀⠀⢀⢀⠀⠀⠀⢀⠀⠀⠀⠀⠀\n⠀⠀⠀⠀⠈⠋⠓⡅⢸⣝⢷⡅⢰⠙⠙⠁⠀⠀⠀⠀\n⠀⢠⣢⣠⡠⣄⠀⡇⢸⢮⡳⡇⢸⠀⡠⡤⡤⡴  O C T O P I I\n⠀⠀⠀⠀⠀⡳⠀⠧⣤⡳⣝⢤⠼⠀⡯⠀⠀⠈⠀ A PII scanner\n⠀⠀⠀⠀⢀⣈⣋⣋⠮⡻⡪⢯⣋⢓⣉⡀    ______________\n⠀⠀⠀⢀⣳⡁⡡⣅⠀⡗⣝⠀⡨⣅⢁⣗⠀⠀  (c) 2023 RedHunt Labs Pvt Ltd\n⠀⠀⠀⠀⠈⠀⠸⣊⣀⡝⢸⣀⣸⠊⠀⠉⠀⠀⠀⠀by Owais Shaikh (owais.shaikh@redhuntlabs.com | me@0x4f.in)\n⠀⠀⠀⠀⠀⠀⠀⠈⠈⠀⠀⠈⠈'
    print(logo)

def help_screen():
    help = 'Usage: python octopii.py <file, local path or URL>\nNote: Only Unix-like filesystems, S3 and open directory URLs are supported.'
    print(help)

def get_regexes():
    with open('definitions.json', 'r', encoding='utf-8') as json_file:
        _rules = json.load(json_file)
        return _rules

def list_local_files(local_path):
    files_list = []
    for root, subdirectories, files in os.walk(local_path):
        for file in files:
            relative_path = os.path.join(root, file)
            files_list.append(relative_path)
    return files_list

def list_directory_files(url):
    urls_list = []
    url = url.replace(' ', '%20')
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(request).read()
    soup = BeautifulSoup(response, 'html.parser')
    a_tags = soup.find_all('a')
    for a_tag in a_tags:
        file_name = ''
        try:
            file_name = re.compile('(?<=<a href=")(.+)(?=">)').findall(str(a_tag))[0]
            if '?C=' in file_name or len(file_name) <= 3:
                raise TypeError
        except TypeError:
            file_name = a_tag.extract().get_text()
        url_new = url + file_name
        url_new = url_new.replace(' ', '%20')
        urls_list.append(url_new)
    return urls_list

def truncate(local_location):
    characters_per_file = 1232500
    file_data = ''
    with open(local_location, 'r') as file:
        file_data = file.read()
        file.close()
    truncated_data = file_data[0:characters_per_file]
    with open(local_location, 'w') as file:
        file.write(truncated_data)
        file.close()

def append_to_output_file(data, file_name):
    try:
        loaded_json = []
        try:
            with open(file_name, 'r+') as read_file:
                loaded_json = json.loads(read_file.read())
        except:
            print("\nCreating new file named '" + file_name + "' and writing to it.")
        with open(file_name, 'w') as write_file:
            loaded_json.append(data)
            write_file.write(json.dumps(loaded_json, indent=4))
    except:
        traceback.print_exc()
        print("Couldn't write to " + file_name + '. Please check if the path is correct and try again.')

def push_data(data: str, url: str):
    headers = {'Content-type': 'application/json'}
    if 'discord' in url:
        payload = {'content': data}
    else:
        payload = {'text': data}
    try:
        req = requests.post(url, headers=headers, json=payload, timeout=7)
        req.raise_for_status()
        print('Scan results sent to webhook.')
    except requests.exceptions.RequestException as e:
        print(f"Couldn't send scan results to webhook. Reason: {e}")


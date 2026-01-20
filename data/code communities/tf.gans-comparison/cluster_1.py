# Cluster 1

def download_celeb_a(dirpath):
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return
    filename, drive_id = ('img_align_celeba.zip', '0B7EVK8r0v71pZjFTYXZWM3FlRnM')
    save_path = os.path.join(dirpath, filename)
    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)
    zip_dir = ''
    with zipfile.ZipFile(save_path) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(save_path)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

def download_file_from_google_drive(id, destination):
    URL = 'https://docs.google.com/uc?export=download'
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def download_lsun(dirpath):
    data_dir = os.path.join(dirpath, 'lsun')
    if os.path.exists(data_dir):
        print('Found LSUN - skip')
        return
    else:
        os.mkdir(data_dir)
    tag = 'latest'
    categories = ['bedroom']
    for category in categories:
        _download_lsun(data_dir, category, 'train', tag)
        _download_lsun(data_dir, category, 'val', tag)
    _download_lsun(data_dir, '', 'test', tag)

def _download_lsun(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}&category={category}&set={set_name}'.format(**locals())
    print(url)
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)

def prepare_data_dir(path='./data'):
    if not os.path.exists(path):
        os.mkdir(path)

def download_mnist(dirpath):
    data_dir = os.path.join(dirpath, 'mnist')
    if os.path.exists(data_dir):
        print('Found MNIST - skip')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base + file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir, file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)


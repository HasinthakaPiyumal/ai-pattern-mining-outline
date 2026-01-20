# Cluster 29

def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):

    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            if curl:
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")
            else:
                torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')
            if delete:
                f.unlink()
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

def download_one(url, dir):
    f = dir / Path(url).name
    if Path(url).is_file():
        Path(url).rename(f)
    elif not f.exists():
        print(f'Downloading {url} to {f}...')
        if curl:
            os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")
        else:
            torch.hub.download_url_to_file(url, f, progress=True)
    if unzip and f.suffix in ('.zip', '.gz'):
        print(f'Unzipping {f}...')
        if f.suffix == '.zip':
            ZipFile(f).extractall(path=dir)
        elif f.suffix == '.gz':
            os.system(f'tar xfz {f} --directory {f.parent}')
        if delete:
            f.unlink()


# Cluster 34

def download_archive_strategy(repo_id: str, output_path: Path, archive_name: Optional[str]) -> bool:
    """Download and extract archive file (fastest method)"""
    print('Using archive download strategy...')
    try:
        if not archive_name:
            archive_name = check_for_archive(repo_id)
            if not archive_name:
                print('No archive found, falling back to parallel downloads')
                return False
        print(f'Downloading archive: {archive_name}')
        archive_path = hf_hub_download(repo_id=repo_id, filename=archive_name, repo_type='dataset', cache_dir=output_path / '.cache')
        print('Extracting sprites...')
        if archive_name.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                members = zf.namelist()
                with tqdm(total=len(members), desc='Extracting') as pbar:
                    for member in members:
                        zf.extract(member, output_path)
                        pbar.update(1)
        elif archive_name.endswith(('.tar.gz', '.tar.bz2', '.tar')):
            mode = 'r:gz' if archive_name.endswith('.gz') else 'r:bz2' if archive_name.endswith('.bz2') else 'r'
            with tarfile.open(archive_path, mode) as tf:
                members = tf.getmembers()
                with tqdm(total=len(members), desc='Extracting') as pbar:
                    for member in members:
                        tf.extract(member, output_path)
                        pbar.update(1)
        else:
            print(f'Unsupported archive format: {archive_name}')
            return False
        cache_dir = output_path / '.cache'
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        print(f'Successfully extracted sprites to {output_path}')
        return True
    except Exception as e:
        print(f'Error with archive strategy: {e}')
        return False

def check_for_archive(repo_id: str) -> Optional[str]:
    """Check if repository contains an archive file with all sprites"""
    try:
        files = list_repo_files(repo_id, repo_type='dataset')
        archive_extensions = ['.tar.gz', '.tar.bz2', '.tar', '.zip', '.7z']
        archives = [f for f in files if any((f.endswith(ext) for ext in archive_extensions))]
        sprite_archives = [a for a in archives if any((keyword in a.lower() for keyword in ['sprite', 'image', 'all', 'complete']))]
        if sprite_archives:
            return max(sprite_archives, key=lambda x: len(x))
        if len(archives) == 1:
            return archives[0]
    except Exception:
        pass
    return None


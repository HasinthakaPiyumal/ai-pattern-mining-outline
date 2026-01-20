# Cluster 26

def flatten_recursive(path='../datasets/coco128'):
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)

def create_folder(path='./new'):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


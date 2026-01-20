# Cluster 10

def download_demo_data(data_dir: str | Path='./dataset') -> Path:
    """
    Download demo data if not exist

    Args:
        data_dir(str | Path): data directory

    Returns:
        pathlib.Path: demo data path
    """
    data_dir = Path(data_dir).expanduser().resolve()
    demo_data_path = data_dir / 'adult.csv'
    if not demo_data_path.exists():
        demo_data_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info('Downloading demo data from github data source to {}'.format(demo_data_path))
        url = 'https://raw.githubusercontent.com/saravrajavelu/Adult-Income-Analysis/master/adult.csv'
        urllib.request.urlretrieve(url, demo_data_path)
    return demo_data_path

def get_demo_single_table(data_dir: str | Path='./dataset'):
    """
    Get demo single table as DataFrame and discrete columns names

    Args:
        data_dir(str | Path): data directory

    Returns:

        pd.DataFrame: demo single table
        list: discrete columns
    """
    demo_data_path = download_demo_data(data_dir)
    pd_obj = pd.read_csv(demo_data_path)
    discrete_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
    return (pd_obj, discrete_cols)

def download_multi_table_demo_data(data_dir: str | Path='./dataset', dataset_name='rossman') -> dict[str, Path]:
    """
    Download multi-table demo data "Rossman Store Sales" or "Rossmann Store Sales" if not exist

    Args:
        data_dir(str | Path): data directory

    Returns:
        dict[str, pathlib.Path]: dict, the key is table name, value is demo data path
    """
    demo_data_info = MULTI_TABLE_DEMO_DATA[dataset_name]
    data_dir = Path(data_dir).expanduser().resolve()
    parent_file_name = dataset_name + '_' + demo_data_info['parent_table'] + '.csv'
    child_file_name = dataset_name + '_' + demo_data_info['child_table'] + '.csv'
    demo_data_path_parent = data_dir / parent_file_name
    demo_data_path_child = data_dir / child_file_name
    if not demo_data_path_parent.exists():
        demo_data_path_parent.parent.mkdir(parents=True, exist_ok=True)
        logger.info('Downloading parent table from github to {}'.format(demo_data_path_parent))
        parent_url = demo_data_info['parent_url']
        urllib.request.urlretrieve(parent_url, demo_data_path_parent)
    if not demo_data_path_child.exists():
        demo_data_path_child.parent.mkdir(parents=True, exist_ok=True)
        logger.info('Downloading child table from github to {}'.format(demo_data_path_child))
        parent_url = demo_data_info['child_url']
        urllib.request.urlretrieve(parent_url, demo_data_path_child)
    return {demo_data_info['parent_table']: demo_data_path_parent, demo_data_info['child_table']: demo_data_path_child}

def get_demo_multi_table(data_dir: str | Path='./dataset', dataset_name='rossman') -> dict[str, pd.DataFrame]:
    """
    Get multi-table demo data as DataFrame and relationship

    Args:
        data_dir(str | Path): data directory

    Returns:
        dict[str, pd.DataFrame]: multi-table data dict, the key is table name, value is DataFrame.
    """
    multi_table_dict = {}
    demo_data_dict = download_multi_table_demo_data(data_dir, dataset_name)
    for table_name in demo_data_dict.keys():
        each_path = demo_data_dict[table_name]
        pd_obj = pd.read_csv(each_path)
        multi_table_dict[table_name] = pd_obj
    return multi_table_dict

@pytest.fixture
def demo_single_table_path():
    yield download_demo_data(DATA_DIR).as_posix()

@pytest.fixture
def demo_multi_table_path():
    yield download_multi_table_demo_data(DATA_DIR)

@pytest.fixture
def single_table_gpt_model():
    yield SingleTableGPTModel()


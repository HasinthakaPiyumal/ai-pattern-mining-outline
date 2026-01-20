# Cluster 67

@pytest.fixture
def dummy_single_table_path(tmp_path):
    dummy_size = 10
    role_set = ['admin', 'user', 'guest']
    df = pd.DataFrame({'role': [random.choice(role_set) for _ in range(dummy_size)], 'name': [ramdon_str() for _ in range(dummy_size)], 'feature_x': [random.random() for _ in range(dummy_size)], 'feature_y': [random.random() for _ in range(dummy_size)], 'feature_z': [random.random() for _ in range(dummy_size)]})
    save_path = tmp_path / 'dummy.csv'
    df.to_csv(save_path, index=False, header=True)
    yield save_path
    save_path.unlink()

def ramdon_str():
    return ''.join((random.choice(string.ascii_letters) for _ in range(10)))

@pytest.fixture
def demo_relational_table_path(tmp_path):
    dummy_size = 10
    role_set = ['admin', 'user', 'guest']
    df = pd.DataFrame({'id': list(range(dummy_size)), 'role': [random.choice(role_set) for _ in range(dummy_size)], 'name': [ramdon_str() for _ in range(dummy_size)], 'feature_x': [random.random() for _ in range(dummy_size)], 'feature_y': [random.random() for _ in range(dummy_size)], 'feature_z': [random.random() for _ in range(dummy_size)]})
    save_path_a = tmp_path / 'dummy_relation_A.csv'
    df.to_csv(save_path_a, index=False, header=True)
    sub_size = 5
    assert dummy_size >= sub_size
    df = pd.DataFrame({'foreign_id': list(range(sub_size)), 'feature_i': [random.random() for _ in range(sub_size)], 'feature_j': [random.random() for _ in range(sub_size)], 'feature_k': [random.random() for _ in range(sub_size)]})
    save_path_b = tmp_path / 'dummy_relation_B.csv'
    df.to_csv(save_path_b, index=False, header=True)
    return (save_path_a, save_path_b, [('id', 'foreign_id')])


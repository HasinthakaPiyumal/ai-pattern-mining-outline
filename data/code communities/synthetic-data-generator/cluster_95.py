# Cluster 95

@pytest.fixture
def test_data_time():
    start_date = '1900-01-01'
    end_date = '2023-12-31'
    df = pd.DataFrame({'time_x': [generate_random_time(start_date, end_date) for _ in range(10)], 'time_y': [generate_random_time(start_date, end_date) for _ in range(10)]})
    return df

def generate_random_time(start_date, end_date):
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    random_time_delta = random.randint(0, int((end_datetime - start_datetime).total_seconds()))
    random_datetime = start_datetime + timedelta(seconds=random_time_delta)
    return random_datetime


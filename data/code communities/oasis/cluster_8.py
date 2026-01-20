# Cluster 8

def generate_user_data(n):
    user_data = []
    start_time = datetime.now()
    max_workers = 100
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_user_profile) for _ in range(n)]
        for i, future in enumerate(as_completed(futures)):
            profile = future.result()
            user_data.append(profile)
            elapsed_time = datetime.now() - start_time
            print(f'Generated {i + 1}/{n} user profiles. Time elapsed: {elapsed_time}')
    return user_data

def save_user_data(user_data, filename):
    with open(filename, 'w') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)


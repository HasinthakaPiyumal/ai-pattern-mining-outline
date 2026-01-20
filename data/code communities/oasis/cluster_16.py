# Cluster 16

def test_rec_sys_personalized_all_posts():
    user_table = [{'user_id': 0, 'bio': 'I like cats'}, {'user_id': 1, 'bio': 'I like dogs'}]
    post_table = [{'post_id': '1', 'user_id': 2, 'content': 'I like dogs'}, {'post_id': '2', 'user_id': 3, 'content': 'I like cats'}]
    trace_table = []
    rec_matrix = [[], []]
    max_rec_post_len = 2
    expected = [['1', '2'], ['1', '2']]
    result = rec_sys_personalized(user_table, post_table, trace_table, rec_matrix, max_rec_post_len)
    assert result == expected

def rec_sys_personalized(user_table: List[Dict[str, Any]], post_table: List[Dict[str, Any]], trace_table: List[Dict[str, Any]], rec_matrix: List[List], max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on personalized similarity scores.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    global model
    if model is None or isinstance(model, tuple):
        model = get_recsys_model(recsys_type='twitter')
    post_ids = [post['post_id'] for post in post_table]
    print(f'Running personalized recommendation for {len(user_table)} users...')
    start_time = time.time()
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        user_bios = [user['bio'] if 'bio' in user and user['bio'] is not None else '' for user in user_table]
        post_contents = [post['content'] for post in post_table]
        if model:
            user_embeddings = model.encode(user_bios, convert_to_tensor=True, device=device)
            post_embeddings = model.encode(post_contents, convert_to_tensor=True, device=device)
            dot_product = torch.matmul(user_embeddings, post_embeddings.T)
            user_norms = torch.norm(user_embeddings, dim=1)
            post_norms = torch.norm(post_embeddings, dim=1)
            similarities = dot_product / (user_norms[:, None] * post_norms[None, :])
        else:
            similarities = torch.rand(len(user_table), len(post_table))
        for user_index, user in enumerate(user_table):
            filtered_post_indices = [i for i, post in enumerate(post_table) if post['user_id'] != user['user_id']]
            user_similarities = similarities[user_index, filtered_post_indices]
            filtered_post_ids = [post_table[i]['post_id'] for i in filtered_post_indices]
            _, top_indices = torch.topk(user_similarities, k=min(max_rec_post_len, len(filtered_post_ids)))
            top_post_ids = [filtered_post_ids[i] for i in top_indices.tolist()]
            new_rec_matrix.append(top_post_ids)
    end_time = time.time()
    print(f'Personalized recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix

def test_rec_sys_personalized_sample_posts():
    user_table = [{'user_id': 0, 'bio': 'I like cats'}, {'user_id': 1, 'bio': 'I like dogs'}]
    post_table = [{'post_id': '1', 'user_id': 2, 'content': 'I like dogs'}, {'post_id': '2', 'user_id': 3, 'content': 'I like cats'}, {'post_id': '3', 'user_id': 4, 'content': 'I like birds'}]
    trace_table = []
    rec_matrix = [[], []]
    max_rec_post_len = 2
    result = rec_sys_personalized(user_table, post_table, trace_table, rec_matrix, max_rec_post_len)
    for rec in result:
        assert len(rec) == max_rec_post_len
        for post_id in rec:
            assert post_id in ['1', '2', '3']
    for i in range(len(result)):
        if i == 0:
            assert result[i] == ['2', '1']
        if i == 1:
            assert result[i] == ['1', '2']

def load_model(model_name):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name == 'paraphrase-MiniLM-L6-v2':
            return SentenceTransformer(model_name, device=device, cache_folder='./models')
        elif model_name == 'Twitter/twhin-bert-base':
            twhin_tokenizer = get_twhin_tokenizer()
            twhin_model = get_twhin_model(device)
            return (twhin_tokenizer, twhin_model)
        else:
            raise ValueError(f'Unknown model name: {model_name}')
    except Exception as e:
        raise Exception(f'Failed to load the model: {model_name}') from e

def get_twhin_tokenizer():
    global twhin_tokenizer
    if twhin_tokenizer is None:
        from transformers import AutoTokenizer
        twhin_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='Twitter/twhin-bert-base', model_max_length=512)
    return twhin_tokenizer

def get_twhin_model(device):
    global twhin_model
    if twhin_model is None:
        from transformers import AutoModel
        twhin_model = AutoModel.from_pretrained(pretrained_model_name_or_path='Twitter/twhin-bert-base').to(device)
    return twhin_model

def get_recsys_model(recsys_type: str=None):
    if recsys_type == RecsysType.TWITTER.value:
        model = load_model('paraphrase-MiniLM-L6-v2')
        return model
    elif recsys_type == RecsysType.TWHIN.value:
        twhin_tokenizer, twhin_model = load_model('Twitter/twhin-bert-base')
        models = (twhin_tokenizer, twhin_model)
        return models
    elif recsys_type == RecsysType.REDDIT.value or recsys_type == RecsysType.RANDOM.value:
        return None
    else:
        raise ValueError(f'Unknown recsys type: {recsys_type}')


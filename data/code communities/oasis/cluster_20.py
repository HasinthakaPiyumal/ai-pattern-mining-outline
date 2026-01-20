# Cluster 20

def test_rec_sys_personalized_twhin():
    user_table = [{'user_id': 0, 'bio': 'I like cats', 'num_followers': 3}, {'user_id': 1, 'bio': 'I like dogs', 'num_followers': 5}, {'user_id': 2, 'bio': '', 'num_followers': 5}, {'user_id': 3, 'bio': '', 'num_followers': 5}]
    post_table = [{'post_id': '1', 'user_id': 2, 'content': 'I like dogs', 'created_at': '0'}, {'post_id': '2', 'user_id': 3, 'content': 'I like cats', 'created_at': '0'}]
    trace_table = []
    rec_matrix = [[], [], [], []]
    max_rec_post_len = 2
    latest_post_count = len(post_table)
    expected = [['1', '2'], ['1', '2'], ['1', '2'], ['1', '2']]
    reset_globals()
    result = rec_sys_personalized_twh(user_table, post_table, latest_post_count, trace_table, rec_matrix, max_rec_post_len, current_time=1)
    assert result == expected

def reset_globals():
    global user_previous_post_all, user_previous_post
    global user_profiles, t_items, u_items
    global date_score
    user_previous_post_all = {}
    user_previous_post = {}
    user_profiles = []
    t_items = {}
    u_items = {}
    date_score = []

def rec_sys_personalized_twh(user_table: List[Dict[str, Any]], post_table: List[Dict[str, Any]], latest_post_count: int, trace_table: List[Dict[str, Any]], rec_matrix: List[List], max_rec_post_len: int, current_time: int, recall_only: bool=False, enable_like_score: bool=False, use_openai_embedding: bool=False) -> List[List]:
    global twhin_model, twhin_tokenizer
    if twhin_model is None or twhin_tokenizer is None:
        twhin_tokenizer, twhin_model = get_recsys_model(recsys_type='twhin-bert')
    global date_score, t_items, u_items, user_previous_post
    global user_previous_post_all, user_profiles
    if not u_items or len(u_items) != len(user_table):
        u_items = {user['user_id']: user['num_followers'] for user in user_table}
    if not user_previous_post_all or len(user_previous_post_all) != len(user_table):
        user_previous_post_all = {index: [] for index in range(len(user_table))}
        user_previous_post = {index: '' for index in range(len(user_table))}
    if not user_profiles or len(user_profiles) != len(user_table):
        for user in user_table:
            if user['bio'] is None:
                user_profiles.append('This user does not have profile')
            else:
                user_profiles.append(user['bio'])
    if len(t_items) < len(post_table):
        for post in post_table[-latest_post_count:]:
            t_items[post['post_id']] = post['content']
            user_previous_post_all[post['user_id']].append(post['content'])
            user_previous_post[post['user_id']] = post['content']
            date_score.append(np.log((271.8 - (current_time - int(post['created_at']))) / 100))
    date_score_np = np.array(date_score)
    if enable_like_score:
        like_post_ids_all = []
        for user in user_table:
            user_id = user['agent_id']
            like_post_ids = get_like_post_id(user_id, ActionType.LIKE_POST.value, trace_table)
            like_post_ids_all.append(like_post_ids)
    scores = date_score_np
    new_rec_matrix = []
    if len(post_table) <= max_rec_post_len:
        tids = [t['post_id'] for t in post_table]
        new_rec_matrix = [tids] * len(rec_matrix)
    else:
        for post_user_index in user_previous_post:
            try:
                update_profile = f' # Recent post:{user_previous_post[post_user_index]}'
                if user_previous_post[post_user_index] != '':
                    if '# Recent post:' not in user_profiles[post_user_index]:
                        user_profiles[post_user_index] += update_profile
                    elif update_profile not in user_profiles[post_user_index]:
                        user_profiles[post_user_index] = user_profiles[post_user_index].split('# Recent post:')[0] + update_profile
            except Exception:
                print('update previous post failed')
        filtered_posts_tuple = coarse_filtering(list(t_items.values()), 4000)
        corpus = user_profiles + filtered_posts_tuple[0]
        tweet_vector_start_t = time.time()
        if use_openai_embedding:
            all_post_vector_list = generate_post_vector_openai(corpus, batch_size=1000)
        else:
            all_post_vector_list = generate_post_vector(twhin_model, twhin_tokenizer, corpus, batch_size=1000)
        tweet_vector_end_t = time.time()
        rec_log.info(f'twhin model cost time: {tweet_vector_end_t - tweet_vector_start_t}')
        user_vector = all_post_vector_list[:len(user_profiles)]
        posts_vector = all_post_vector_list[len(user_profiles):]
        if enable_like_score:
            like_posts_vectors = []
            for user_idx, like_post_ids in enumerate(like_post_ids_all):
                if len(like_post_ids) != 1:
                    for like_post_id in like_post_ids:
                        try:
                            like_posts_vectors.append(posts_vector[like_post_id - 1])
                        except Exception:
                            like_posts_vectors.append(user_vector[user_idx])
                else:
                    like_posts_vectors += [user_vector[user_idx] for _ in range(5)]
            try:
                like_posts_vectors = torch.stack(like_posts_vectors).view(len(user_table), 5, posts_vector.shape[1])
            except Exception:
                import pdb
                pdb.set_trace()
        get_similar_start_t = time.time()
        cosine_similarities = cosine_similarity(user_vector, posts_vector)
        get_similar_end_t = time.time()
        rec_log.info(f'get cosine_similarity time: {get_similar_end_t - get_similar_start_t}')
        if enable_like_score:
            for user_index, profile in enumerate(user_profiles):
                user_like_posts_vector = like_posts_vectors[user_index]
                like_scores = calculate_like_similarity(user_like_posts_vector, posts_vector)
                try:
                    scores = scores + like_scores
                except Exception:
                    import pdb
                    pdb.set_trace()
        filter_posts_index = filtered_posts_tuple[1]
        cosine_similarities = cosine_similarities * scores[filter_posts_index]
        cosine_similarities = torch.tensor(cosine_similarities)
        value, indices = torch.topk(cosine_similarities, max_rec_post_len, dim=1, largest=True, sorted=True)
        filter_posts_index = torch.tensor(filter_posts_index)
        indices = filter_posts_index[indices]
        matrix_list = indices.cpu().numpy()
        post_list = list(t_items.keys())
        for rec_ids in matrix_list:
            rec_ids = [post_list[i] for i in rec_ids]
            new_rec_matrix.append(rec_ids)
    return new_rec_matrix

def test_rec_sys_personalized_twhin_sample_posts():
    user_table = [{'user_id': 0, 'bio': 'I like cats', 'num_followers': 3}, {'user_id': 1, 'bio': 'I like dogs', 'num_followers': 3}, {'user_id': 2, 'bio': '', 'num_followers': 3}, {'user_id': 3, 'bio': '', 'num_followers': 3}, {'user_id': 4, 'bio': '', 'num_followers': 3}]
    post_table = [{'post_id': '1', 'user_id': 2, 'content': 'I like dogs', 'created_at': '0'}, {'post_id': '2', 'user_id': 3, 'content': 'I like cats', 'created_at': '0'}, {'post_id': '3', 'user_id': 4, 'content': 'I like birds', 'created_at': '0'}]
    trace_table = []
    rec_matrix = [[], [], [], [], []]
    max_rec_post_len = 2
    latest_post_count = len(post_table)
    reset_globals()
    result = rec_sys_personalized_twh(user_table, post_table, latest_post_count, trace_table, rec_matrix, max_rec_post_len, current_time=1)
    for rec in result:
        assert len(rec) == max_rec_post_len
        for post_id in rec:
            assert post_id in ['1', '2', '3']
    for i in range(len(result)):
        if i == 0:
            assert result[i] == ['2', '1']
        if i == 1:
            assert result[i] == ['1', '2']

def get_like_post_id(user_id, action, trace_table):
    """
    Get the post IDs that a user has liked or unliked.

    Args:
        user_id (str): ID of the user.
        action (str): Type of action (like or unlike).
        post_table (list): List of posts.
        trace_table (list): List of user interactions.

    Returns:
        list: List of post IDs.
    """
    trace_post_ids = [literal_eval(trace['info'])['post_id'] for trace in trace_table if trace['user_id'] == user_id and trace['action'] == action]
    'Only take the last 5 liked posts, if not enough, pad with the most\n    recently liked post. Only take IDs, not content, because calculating\n    embeddings for all posts again is very time-consuming, especially when the\n    number of agents is large'
    if len(trace_post_ids) < 5 and len(trace_post_ids) > 0:
        trace_post_ids += [trace_post_ids[-1]] * (5 - len(trace_post_ids))
    elif len(trace_post_ids) > 5:
        trace_post_ids = trace_post_ids[-5:]
    else:
        trace_post_ids = [0]
    return trace_post_ids

def coarse_filtering(input_list, scale):
    """
    Coarse filtering posts and return selected elements with their indices.
    """
    if len(input_list) <= scale:
        sampled_indices = range(len(input_list))
        return (input_list, sampled_indices)
    else:
        sampled_indices = random.sample(range(len(input_list)), scale)
        sampled_elements = [input_list[idx] for idx in sampled_indices]
        return (sampled_elements, sampled_indices)

def calculate_like_similarity(liked_vectors, target_vectors):
    liked_norms = np.linalg.norm(liked_vectors, axis=1)
    target_norms = np.linalg.norm(target_vectors, axis=1)
    dot_products = np.dot(target_vectors, liked_vectors.T)
    cosine_similarities = dot_products / np.outer(target_norms, liked_norms)
    average_similarities = np.mean(cosine_similarities, axis=1)
    return average_similarities


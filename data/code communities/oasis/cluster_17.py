# Cluster 17

def rec_sys_personalized_with_trace(user_table: List[Dict[str, Any]], post_table: List[Dict[str, Any]], trace_table: List[Dict[str, Any]], rec_matrix: List[List], max_rec_post_len: int, swap_rate: float=0.1) -> List[List]:
    """
    This version:
    1. If the number of posts is less than or equal to the maximum
        recommended length, each user gets all post IDs

    2. Otherwise:
        - For each user, get a like-trace pool and dislike-trace pool from the
            trace table
        - For each user, calculate the similarity between the user's bio and
            the post text
        - Use the trace table to adjust the similarity score
        - Swap 10% of the recommended posts with the random posts

    Personalized recommendation system that uses user interaction traces.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.
        swap_rate (float): Percentage of posts to swap for diversity.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    start_time = time.time()
    new_rec_matrix = []
    post_ids = [post['post_id'] for post in post_table]
    if len(post_ids) <= max_rec_post_len:
        new_rec_matrix = [post_ids] * (len(rec_matrix) - 1)
    else:
        for idx in range(1, len(rec_matrix)):
            user_id = user_table[idx - 1]['user_id']
            user_bio = user_table[idx - 1]['bio']
            available_post_contents = [(post['post_id'], post['content']) for post in post_table if post['user_id'] != user_id]
            like_trace_contents = get_trace_contents(user_id, ActionType.LIKE_POST.value, post_table, trace_table)
            dislike_trace_contents = get_trace_contents(user_id, ActionType.UNLIKE_POST.value, post_table, trace_table)
            post_scores = []
            for post_id, post_content in available_post_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    post_embedding = model.encode(post_content)
                    base_similarity = np.dot(user_embedding, post_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(post_embedding))
                    post_scores.append((post_id, base_similarity))
                else:
                    post_scores.append((post_id, random.random()))
            new_post_scores = []
            for _post_id, _base_similarity in post_scores:
                _post_content = post_table[post_ids.index(_post_id)]['content']
                like_similarity = sum((np.dot(model.encode(_post_content), model.encode(like)) / (np.linalg.norm(model.encode(_post_content)) * np.linalg.norm(model.encode(like))) for like in like_trace_contents)) / len(like_trace_contents) if like_trace_contents else 0
                dislike_similarity = sum((np.dot(model.encode(_post_content), model.encode(dislike)) / (np.linalg.norm(model.encode(_post_content)) * np.linalg.norm(model.encode(dislike))) for dislike in dislike_trace_contents)) / len(dislike_trace_contents) if dislike_trace_contents else 0
                adjusted_similarity = normalize_similarity_adjustments(post_scores, _base_similarity, like_similarity, dislike_similarity)
                new_post_scores.append((_post_id, adjusted_similarity))
            new_post_scores.sort(key=lambda x: x[1], reverse=True)
            rec_post_ids = [post_id for post_id, _ in new_post_scores[:max_rec_post_len]]
            if swap_rate > 0:
                swap_free_ids = [post_id for post_id in post_ids if post_id not in rec_post_ids and post_id not in [trace['post_id'] for trace in trace_table if trace['user_id']]]
                rec_post_ids = swap_random_posts(rec_post_ids, swap_free_ids, swap_rate)
            new_rec_matrix.append(rec_post_ids)
    end_time = time.time()
    print(f'Personalized recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix

def get_trace_contents(user_id, action, post_table, trace_table):
    """
    Get the contents of posts that a user has interacted with.

    Args:
        user_id (str): ID of the user.
        action (str): Type of action (like or unlike).
        post_table (list): List of posts.
        trace_table (list): List of user interactions.

    Returns:
        list: List of post contents.
    """
    trace_post_ids = [trace['post_id'] for trace in trace_table if trace['user_id'] == user_id and trace['action'] == action]
    trace_contents = [post['content'] for post in post_table if post['post_id'] in trace_post_ids]
    return trace_contents

def normalize_similarity_adjustments(post_scores, base_similarity, like_similarity, dislike_similarity):
    """
    Normalize the adjustments to keep them in scale with overall similarities.

    Args:
        post_scores (list): List of post scores.
        base_similarity (float): Base similarity score.
        like_similarity (float): Similarity score for liked posts.
        dislike_similarity (float): Similarity score for disliked posts.

    Returns:
        float: Adjusted similarity score.
    """
    if len(post_scores) == 0:
        return base_similarity
    max_score = max(post_scores, key=lambda x: x[1])[1]
    min_score = min(post_scores, key=lambda x: x[1])[1]
    score_range = max_score - min_score
    adjustment = (like_similarity - dislike_similarity) * (score_range / 2)
    return base_similarity + adjustment

def swap_random_posts(rec_post_ids, post_ids, swap_percent=0.1):
    """
    Swap a percentage of recommended posts with random posts.

    Args:
        rec_post_ids (list): List of recommended post IDs.
        post_ids (list): List of all post IDs.
        swap_percent (float): Percentage of posts to swap.

    Returns:
        list: Updated list of recommended post IDs.
    """
    num_to_swap = int(len(rec_post_ids) * swap_percent)
    posts_to_swap = random.sample(post_ids, num_to_swap)
    indices_to_replace = random.sample(range(len(rec_post_ids)), num_to_swap)
    for idx, new_post in zip(indices_to_replace, posts_to_swap):
        rec_post_ids[idx] = new_post
    return rec_post_ids


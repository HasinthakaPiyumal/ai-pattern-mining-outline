# Cluster 19

def test_rec_sys_random_all_posts():
    post_table = [{'post_id': '1'}, {'post_id': '2'}]
    rec_matrix = [[], []]
    max_rec_post_len = 2
    expected = [['1', '2'], ['1', '2']]
    result = rec_sys_random(post_table, rec_matrix, max_rec_post_len)
    assert result == expected

def rec_sys_random(post_table: List[Dict[str, Any]], rec_matrix: List[List], max_rec_post_len: int) -> List[List]:
    """
    Randomly recommend posts to users.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    post_ids = [post['post_id'] for post in post_table]
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        for _ in range(len(rec_matrix)):
            new_rec_matrix.append(random.sample(post_ids, max_rec_post_len))
    return new_rec_matrix

def test_rec_sys_reddit_all_posts():
    post_table = [{'post_id': '1'}, {'post_id': '2'}]
    rec_matrix = [[], []]
    max_rec_post_len = 2
    expected = [['1', '2'], ['1', '2']]
    result = rec_sys_reddit(post_table, rec_matrix, max_rec_post_len)
    assert result == expected

def rec_sys_reddit(post_table: List[Dict[str, Any]], rec_matrix: List[List], max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on Reddit-like hot score.

    Args:
        post_table (List[Dict[str, Any]]): List of posts.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    post_ids = [post['post_id'] for post in post_table]
    if len(post_ids) <= max_rec_post_len:
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        all_hot_score = []
        for post in post_table:
            try:
                created_at_dt = datetime.strptime(post['created_at'], '%Y-%m-%d %H:%M:%S.%f')
            except Exception:
                created_at_dt = datetime.strptime(post['created_at'], '%Y-%m-%d %H:%M:%S')
            hot_score = calculate_hot_score(post['num_likes'], post['num_dislikes'], created_at_dt)
            all_hot_score.append((hot_score, post['post_id']))
        top_posts = heapq.nlargest(max_rec_post_len, all_hot_score, key=lambda x: x[0])
        top_post_ids = [post_id for _, post_id in top_posts]
        new_rec_matrix = [top_post_ids] * len(rec_matrix)
    return new_rec_matrix

def test_rec_sys_random_sample_posts():
    post_table = [{'post_id': '1'}, {'post_id': '2'}, {'post_id': '3'}]
    rec_matrix = [[], []]
    max_rec_post_len = 2
    result = rec_sys_random(post_table, rec_matrix, max_rec_post_len)
    for rec in result:
        assert len(rec) == max_rec_post_len
        for post_id in rec:
            assert post_id in ['1', '2', '3']

def test_rec_sys_reddit_sample_posts():
    post_table = [{'post_id': '1', 'num_likes': 100000, 'num_dislikes': 25, 'created_at': '2024-06-25 12:00:00.222000'}, {'post_id': '2', 'num_likes': 90, 'num_dislikes': 30, 'created_at': '2024-06-26 12:00:00.321009'}, {'post_id': '3', 'num_likes': 75, 'num_dislikes': 50, 'created_at': '2024-06-27 12:00:00.123009'}, {'post_id': '4', 'num_likes': 70, 'num_dislikes': 50, 'created_at': '2024-06-27 13:00:00.321009'}]
    rec_matrix = [[], []]
    max_rec_post_len = 3
    result = rec_sys_reddit(post_table, rec_matrix, max_rec_post_len)
    for rec in result:
        assert len(rec) == max_rec_post_len
        for post_id in rec:
            assert post_id in ['3', '4', '1']

def test_user_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO user (agent_id, user_name, name, bio, created_at, num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)', (2, 'testuser', 'Test User', 'A test user', '2024-04-21 22:02:42', 0, 0))
    conn.commit()
    cursor.execute("SELECT * FROM user WHERE user_name = 'testuser'")
    user = cursor.fetchone()
    assert user is not None
    assert user[1] == 2
    assert user[2] == 'testuser'
    assert user[3] == 'Test User'
    assert user[4] == 'A test user'
    assert user[5] == '2024-04-21 22:02:42'
    assert user[6] == 0
    assert user[7] == 0
    cursor.execute('UPDATE user SET name = ? WHERE user_name = ?', ('Updated User', 'testuser'))
    conn.commit()
    cursor.execute("SELECT * FROM user WHERE user_name = 'testuser'")
    user = cursor.fetchone()
    assert user[3] == 'Updated User'
    cursor.execute('INSERT INTO user (agent_id, user_name, name, bio, created_at, num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)', (1, 'testuser_2', 'Test User_2', 'Another user', '2024-05-21 22:02:42', 0, 0))
    conn.commit()
    expected_result = [{'user_id': 1, 'agent_id': 2, 'user_name': 'testuser', 'name': 'Updated User', 'bio': 'A test user', 'created_at': '2024-04-21 22:02:42', 'num_followings': 0, 'num_followers': 0}, {'user_id': 2, 'agent_id': 1, 'user_name': 'testuser_2', 'name': 'Test User_2', 'bio': 'Another user', 'created_at': '2024-05-21 22:02:42', 'num_followings': 0, 'num_followers': 0}]
    actual_result = fetch_table_from_db(cursor, 'user')
    assert actual_result == expected_result, 'The fetched data does not match.'
    cursor.execute('INSERT INTO user (agent_id, user_name, name, bio, created_at, num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)', (3, 'testuser_3', 'Test User_3', 'Third user', '2024-05-21 22:02:42', 0, 0))
    conn.commit()
    cursor.execute("DELETE FROM user WHERE user_name = 'testuser_3'")
    conn.commit()
    cursor.execute("SELECT * FROM user WHERE user_name = 'testuser_3'")
    assert cursor.fetchone() is None

def fetch_table_from_db(cursor: sqlite3.Cursor, table_name: str) -> List[Dict[str, Any]]:
    cursor.execute(f'SELECT * FROM {table_name}')
    columns = [description[0] for description in cursor.description]
    data_dicts = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return data_dicts

def test_post_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO post (user_id, content, created_at, num_likes, num_dislikes, num_shares, num_reports) VALUES (?, ?, ?, ?, ?, ?, ?)', (1, 'This is a test post', '2024-04-21 22:02:42', 0, 1, 2, 0))
    conn.commit()
    cursor.execute("SELECT * FROM post WHERE content = 'This is a test post'")
    post = cursor.fetchone()
    assert post is not None
    assert post[1] == 1
    assert post[3] == 'This is a test post'
    assert post[5] == '2024-04-21 22:02:42'
    assert post[6] == 0
    assert post[7] == 1
    assert post[8] == 2
    assert post[9] == 0
    cursor.execute('UPDATE post SET content = ? WHERE content = ?', ('Updated post', 'This is a test post'))
    conn.commit()
    expected_result = [{'post_id': 1, 'user_id': 1, 'original_post_id': None, 'content': 'Updated post', 'quote_content': None, 'created_at': '2024-04-21 22:02:42', 'num_likes': 0, 'num_dislikes': 1, 'num_shares': 2, 'num_reports': 0}]
    actual_result = fetch_table_from_db(cursor, 'post')
    assert actual_result == expected_result, 'The fetched data does not match.'
    cursor.execute("SELECT * FROM post WHERE content = 'Updated post'")
    post = cursor.fetchone()
    assert post[3] == 'Updated post'
    cursor.execute("DELETE FROM post WHERE content = 'Updated post'")
    conn.commit()
    cursor.execute("SELECT * FROM post WHERE content = 'Updated post'")
    assert cursor.fetchone() is None

def test_trace_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    created_at = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    cursor.execute('INSERT INTO trace (user_id, created_at, action, info) VALUES (?, ?, ?, ?)', (1, created_at, 'test_action', 'test_info'))
    conn.commit()
    cursor.execute('SELECT * FROM trace WHERE user_id = 1 AND created_at = ?', (created_at,))
    trace = cursor.fetchone()
    assert trace is not None
    assert trace[0] == 1
    assert trace[1] == created_at
    assert trace[2] == 'test_action'
    assert trace[3] == 'test_info'
    expected_result = [{'user_id': 1, 'created_at': created_at, 'action': 'test_action', 'info': 'test_info'}]
    actual_result = fetch_table_from_db(cursor, 'trace')
    assert actual_result == expected_result
    cursor.execute('DELETE FROM trace WHERE user_id = 1 AND created_at = ?', (created_at,))
    conn.commit()
    cursor.execute('SELECT * FROM trace WHERE user_id = 1 AND created_at = ?', (created_at,))
    assert cursor.fetchone() is None

def test_rec_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO rec (user_id, post_id) VALUES (?, ?)', (2, 2))
    cursor.execute('INSERT INTO rec (user_id, post_id) VALUES (?, ?)', (2, 3))
    cursor.execute('INSERT INTO rec (user_id, post_id) VALUES (?, ?)', (1, 3))
    conn.commit()
    cursor.execute('SELECT * FROM rec WHERE user_id = ? AND post_id = ?', (2, 2))
    record = cursor.fetchone()
    assert record is not None
    assert record[0] == 2
    assert record[1] == 2
    cursor.execute('SELECT * FROM rec WHERE user_id = ? AND post_id = ?', (2, 3))
    record = cursor.fetchone()
    assert record is not None
    assert record[0] == 2
    assert record[1] == 3
    assert fetch_rec_table_as_matrix(cursor) == [[3], [2, 3]]
    cursor.execute('DELETE FROM rec WHERE user_id = 2 AND post_id = 2')
    conn.commit()
    cursor.execute('SELECT * FROM rec WHERE user_id = 2 AND post_id = 2')
    assert cursor.fetchone() is None

def fetch_rec_table_as_matrix(cursor: sqlite3.Cursor) -> List[List[int]]:
    cursor.execute('SELECT user_id FROM user ORDER BY user_id')
    user_ids = [row[0] for row in cursor.fetchall()]
    cursor.execute('SELECT user_id, post_id FROM rec ORDER BY user_id, post_id')
    rec_rows = cursor.fetchall()
    user_posts = {user_id: [] for user_id in user_ids}
    for user_id, post_id in rec_rows:
        if user_id in user_posts:
            user_posts[user_id].append(post_id)
    matrix = [user_posts[user_id] for user_id in user_ids]
    return matrix

def test_comment_operations():
    conn = sqlite3.connect(db_filepath)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO comment (post_id, user_id, content, created_at) VALUES (?, ?, ?, ?)', (1, 2, 'This is a test comment', '2024-04-21 22:05:00'))
    conn.commit()
    cursor.execute("SELECT * FROM comment WHERE content = 'This is a test comment'")
    comment = cursor.fetchone()
    assert comment is not None, 'Comment insertion failed.'
    assert comment[1] == 1, 'Post ID mismatch.'
    assert comment[2] == 2, 'User ID mismatch.'
    assert comment[3] == 'This is a test comment', 'Content mismatch.'
    assert comment[4] == '2024-04-21 22:05:00', 'Created at mismatch.'
    assert comment[5] == 0, 'Likes count mismatch.'
    assert comment[6] == 0, 'Dislikes count mismatch.'
    cursor.execute('UPDATE comment SET content = ? WHERE content = ?', ('Updated comment', 'This is a test comment'))
    conn.commit()
    expected_result = [{'comment_id': 1, 'post_id': 1, 'user_id': 2, 'content': 'Updated comment', 'created_at': '2024-04-21 22:05:00', 'num_likes': 0, 'num_dislikes': 0}]
    actual_result = fetch_table_from_db(cursor, 'comment')
    assert actual_result == expected_result, 'The fetched data does not match.'
    cursor.execute("SELECT * FROM comment WHERE content = 'Updated comment'")
    comment = cursor.fetchone()
    assert comment[3] == 'Updated comment', 'Comment update failed.'
    cursor.execute("DELETE FROM comment WHERE content = 'Updated comment'")
    conn.commit()
    cursor.execute("SELECT * FROM comment WHERE content = 'Updated comment'")
    assert cursor.fetchone() is None, 'Comment deletion failed.'

def calculate_hot_score(num_likes: int, num_dislikes: int, created_at: datetime) -> int:
    """
    Compute the hot score for a post.

    Args:
        num_likes (int): Number of likes.
        num_dislikes (int): Number of dislikes.
        created_at (datetime): Creation time of the post.

    Returns:
        int: Hot score of the post.

    Reference:
        https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9
    """
    s = num_likes - num_dislikes
    order = log(max(abs(s), 1), 10)
    sign = 1 if s > 0 else -1 if s < 0 else 0
    epoch = datetime(1970, 1, 1)
    td = created_at - epoch
    epoch_seconds_result = td.days * 86400 + td.seconds + float(td.microseconds) / 1000000.0
    seconds = epoch_seconds_result - 1134028003
    return round(sign * order + seconds / 45000, 7)


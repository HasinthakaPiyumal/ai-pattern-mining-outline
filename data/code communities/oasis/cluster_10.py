# Cluster 10

@pytest.fixture(scope='session', autouse=True)
def setup_session():
    create_db(db_filepath)
    yield
    if osp.exists(db_filepath):
        os.remove(db_filepath)

def create_db(db_path: str | None=None):
    """Create the database if it does not exist. A :obj:`twitter.db`
    file will be automatically created  in the :obj:`data` directory.
    """
    schema_dir = get_schema_dir_path()
    if db_path is None:
        db_path = get_db_path()
    print('db_path', db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        user_sql_path = osp.join(schema_dir, USER_SCHEMA_SQL)
        with open(user_sql_path, 'r') as sql_file:
            user_sql_script = sql_file.read()
        cursor.executescript(user_sql_script)
        post_sql_path = osp.join(schema_dir, POST_SCHEMA_SQL)
        with open(post_sql_path, 'r') as sql_file:
            post_sql_script = sql_file.read()
        cursor.executescript(post_sql_script)
        follow_sql_path = osp.join(schema_dir, FOLLOW_SCHEMA_SQL)
        with open(follow_sql_path, 'r') as sql_file:
            follow_sql_script = sql_file.read()
        cursor.executescript(follow_sql_script)
        mute_sql_path = osp.join(schema_dir, MUTE_SCHEMA_SQL)
        with open(mute_sql_path, 'r') as sql_file:
            mute_sql_script = sql_file.read()
        cursor.executescript(mute_sql_script)
        like_sql_path = osp.join(schema_dir, LIKE_SCHEMA_SQL)
        with open(like_sql_path, 'r') as sql_file:
            like_sql_script = sql_file.read()
        cursor.executescript(like_sql_script)
        dislike_sql_path = osp.join(schema_dir, DISLIKE_SCHEMA_SQL)
        with open(dislike_sql_path, 'r') as sql_file:
            dislike_sql_script = sql_file.read()
        cursor.executescript(dislike_sql_script)
        report_sql_path = osp.join(schema_dir, REPORT_SCHEAM_SQL)
        with open(report_sql_path, 'r') as sql_file:
            report_sql_script = sql_file.read()
        cursor.executescript(report_sql_script)
        trace_sql_path = osp.join(schema_dir, TRACE_SCHEMA_SQL)
        with open(trace_sql_path, 'r') as sql_file:
            trace_sql_script = sql_file.read()
        cursor.executescript(trace_sql_script)
        rec_sql_path = osp.join(schema_dir, REC_SCHEMA_SQL)
        with open(rec_sql_path, 'r') as sql_file:
            rec_sql_script = sql_file.read()
        cursor.executescript(rec_sql_script)
        comment_sql_path = osp.join(schema_dir, COMMENT_SCHEMA_SQL)
        with open(comment_sql_path, 'r') as sql_file:
            comment_sql_script = sql_file.read()
        cursor.executescript(comment_sql_script)
        comment_like_sql_path = osp.join(schema_dir, COMMENT_LIKE_SCHEMA_SQL)
        with open(comment_like_sql_path, 'r') as sql_file:
            comment_like_sql_script = sql_file.read()
        cursor.executescript(comment_like_sql_script)
        comment_dislike_sql_path = osp.join(schema_dir, COMMENT_DISLIKE_SCHEMA_SQL)
        with open(comment_dislike_sql_path, 'r') as sql_file:
            comment_dislike_sql_script = sql_file.read()
        cursor.executescript(comment_dislike_sql_script)
        product_sql_path = osp.join(schema_dir, PRODUCT_SCHEMA_SQL)
        with open(product_sql_path, 'r') as sql_file:
            product_sql_script = sql_file.read()
        cursor.executescript(product_sql_script)
        group_sql_path = osp.join(schema_dir, GROUP_SCHEMA_SQL)
        with open(group_sql_path, 'r') as sql_file:
            group_sql_script = sql_file.read()
        cursor.executescript(group_sql_script)
        group_member_sql_path = osp.join(schema_dir, GROUP_MEMBER_SCHEMA_SQL)
        with open(group_member_sql_path, 'r') as sql_file:
            group_member_sql_script = sql_file.read()
        cursor.executescript(group_member_sql_script)
        group_message_sql_path = osp.join(schema_dir, GROUP_MESSAGE_SCHEMA_SQL)
        with open(group_message_sql_path, 'r') as sql_file:
            group_message_sql_script = sql_file.read()
        cursor.executescript(group_message_sql_script)
        conn.commit()
    except sqlite3.Error as e:
        print(f'An error occurred while creating tables: {e}')
    return (conn, cursor)

def test_multi_signup():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)
    N = 100
    create_db(test_db_filepath)
    db = sqlite3.connect(test_db_filepath, check_same_thread=False)
    db_cursor = db.cursor()
    user_insert_query = 'INSERT INTO user (agent_id, user_name, name, bio, created_at, num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?)'
    for i in range(N):
        db_cursor.execute(user_insert_query, (i, i, i, i, datetime.now(), 0, 0))
        db.commit()
    db_cursor.execute('SELECT * FROM user')
    users = db_cursor.fetchall()
    assert len(users) == N

def get_db_path() -> str:
    env_db_path = os.environ.get('OASIS_DB_PATH')
    if env_db_path:
        return env_db_path
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    db_dir = osp.join(parent_dir, DB_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db_path = osp.join(db_dir, DB_NAME)
    return db_path

def get_schema_dir_path() -> str:
    curr_file_path = osp.abspath(__file__)
    parent_dir = osp.dirname(osp.dirname(curr_file_path))
    schema_dir = osp.join(parent_dir, SCHEMA_DIR)
    return schema_dir

def print_db_tables_summary():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        if table_name not in TABLE_NAMES:
            continue
        print(f'Table: {table_name}')
        cursor.execute(f'PRAGMA table_info({table_name})')
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        print('- Columns:', column_names)
        cursor.execute(f'PRAGMA foreign_key_list({table_name})')
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            print('- Foreign Keys:')
            for fk in foreign_keys:
                print(f'    {fk[2]} references {fk[3]}({fk[4]}) on update {fk[5]} on delete {fk[6]}')
        else:
            print('  No foreign keys.')
        cursor.execute(f'SELECT * FROM {table_name} LIMIT 5;')
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        print()
    conn.close()

class Platform:
    """Platform."""

    def __init__(self, db_path: str, channel: Any=None, sandbox_clock: Clock | None=None, start_time: datetime | None=None, show_score: bool=False, allow_self_rating: bool=True, recsys_type: str | RecsysType='reddit', refresh_rec_post_count: int=1, max_rec_post_len: int=2, following_post_count=3, use_openai_embedding: bool=False):
        self.db_path = db_path
        self.recsys_type = recsys_type
        if sandbox_clock is None:
            sandbox_clock = Clock(60)
        if start_time is None:
            start_time = datetime.now()
        self.start_time = start_time
        self.sandbox_clock = sandbox_clock
        self.db, self.db_cursor = create_db(self.db_path)
        self.db.execute('PRAGMA synchronous = OFF')
        self.channel = channel or Channel()
        self.recsys_type = RecsysType(recsys_type)
        self.show_score = show_score
        self.allow_self_rating = allow_self_rating
        self.refresh_rec_post_count = refresh_rec_post_count
        self.following_post_count = following_post_count
        self.max_rec_post_len = max_rec_post_len
        self.rec_prob = 0.7
        self.use_openai_embedding = use_openai_embedding
        self.trend_num_days = 7
        self.trend_top_k = 1
        self.report_threshold = 2
        self.pl_utils = PlatformUtils(self.db, self.db_cursor, self.start_time, self.sandbox_clock, self.show_score, self.recsys_type, self.report_threshold)

    async def running(self):
        while True:
            message_id, data = await self.channel.receive_from()
            agent_id, message, action = data
            action = ActionType(action)
            if action == ActionType.EXIT:
                if self.db_path == ':memory:':
                    dst = sqlite3.connect('mock.db')
                    with dst:
                        self.db.backup(dst)
                self.db_cursor.close()
                self.db.close()
                break
            action_function = getattr(self, action.value, None)
            if action_function:
                func_code = action_function.__code__
                param_names = func_code.co_varnames[:func_code.co_argcount]
                len_param_names = len(param_names)
                if len_param_names > 3:
                    raise ValueError(f'Functions with {len_param_names} parameters are not supported.')
                params = {}
                if len_param_names >= 2:
                    params['agent_id'] = agent_id
                if len_param_names == 3:
                    second_param_name = param_names[2]
                    params[second_param_name] = message
                result = await action_function(**params)
                await self.channel.send_to((message_id, agent_id, result))
            else:
                raise ValueError(f'Action {action} is not supported')

    def run(self):
        asyncio.run(self.running())

    async def sign_up(self, agent_id, user_message):
        user_name, name, bio = user_message
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_insert_query = 'INSERT INTO user (user_id, agent_id, user_name, name, bio, created_at, num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
            self.pl_utils._execute_db_command(user_insert_query, (agent_id, agent_id, user_name, name, bio, current_time, 0, 0), commit=True)
            user_id = agent_id
            action_info = {'name': name, 'user_name': user_name, 'bio': bio}
            self.pl_utils._record_trace(user_id, ActionType.SIGNUP.value, action_info, current_time)
            return {'success': True, 'user_id': user_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def sign_up_product(self, product_id: int, product_name: str):
        try:
            product_insert_query = 'INSERT INTO product (product_id, product_name) VALUES (?, ?)'
            self.pl_utils._execute_db_command(product_insert_query, (product_id, product_name), commit=True)
            return {'success': True, 'product_id': product_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def purchase_product(self, agent_id, purchase_message):
        product_name, purchase_num = purchase_message
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        user_id = agent_id
        product_check_query = "SELECT * FROM 'product' WHERE product_name = ?"
        self.pl_utils._execute_db_command(product_check_query, (product_name,))
        check_result = self.db_cursor.fetchone()
        if not check_result:
            return {'success': False, 'error': 'No such product.'}
        else:
            product_id = check_result[0]
        product_update_query = 'UPDATE product SET sales = sales + ? WHERE product_name = ?'
        self.pl_utils._execute_db_command(product_update_query, (purchase_num, product_name), commit=True)
        action_info = {'product_name': product_name, 'purchase_num': purchase_num}
        self.pl_utils._record_trace(user_id, ActionType.PURCHASE_PRODUCT.value, action_info, current_time)
        return {'success': True, 'product_id': product_id}

    async def refresh(self, agent_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            rec_query = 'SELECT post_id FROM rec WHERE user_id = ?'
            self.pl_utils._execute_db_command(rec_query, (user_id,))
            rec_results = self.db_cursor.fetchall()
            post_ids = [row[0] for row in rec_results]
            selected_post_ids = post_ids
            if len(selected_post_ids) >= self.refresh_rec_post_count:
                selected_post_ids = random.sample(selected_post_ids, self.refresh_rec_post_count)
            if self.recsys_type != RecsysType.REDDIT:
                query_following_post = 'SELECT post.post_id, post.user_id, post.content, post.created_at, post.num_likes FROM post JOIN follow ON post.user_id = follow.followee_id WHERE follow.follower_id = ? ORDER BY post.num_likes DESC LIMIT ?'
                self.pl_utils._execute_db_command(query_following_post, (user_id, self.following_post_count))
                following_posts = self.db_cursor.fetchall()
                following_posts_ids = [row[0] for row in following_posts]
                selected_post_ids = following_posts_ids + selected_post_ids
                selected_post_ids = list(set(selected_post_ids))
            placeholders = ', '.join(('?' for _ in selected_post_ids))
            post_query = f'SELECT post_id, user_id, original_post_id, content, quote_content, created_at, num_likes, num_dislikes, num_shares FROM post WHERE post_id IN ({placeholders})'
            self.pl_utils._execute_db_command(post_query, selected_post_ids)
            results = self.db_cursor.fetchall()
            if not results:
                return {'success': False, 'message': 'No posts found.'}
            results_with_comments = self.pl_utils._add_comments_to_posts(results)
            action_info = {'posts': results_with_comments}
            self.pl_utils._record_trace(user_id, ActionType.REFRESH.value, action_info, current_time)
            return {'success': True, 'posts': results_with_comments}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def update_rec_table(self):
        twitter_log.info('Starting to refresh recommendation system cache...')
        user_table = fetch_table_from_db(self.db_cursor, 'user')
        post_table = fetch_table_from_db(self.db_cursor, 'post')
        trace_table = fetch_table_from_db(self.db_cursor, 'trace')
        rec_matrix = fetch_rec_table_as_matrix(self.db_cursor)
        if self.recsys_type == RecsysType.RANDOM:
            new_rec_matrix = rec_sys_random(post_table, rec_matrix, self.max_rec_post_len)
        elif self.recsys_type == RecsysType.TWITTER:
            new_rec_matrix = rec_sys_personalized_with_trace(user_table, post_table, trace_table, rec_matrix, self.max_rec_post_len)
        elif self.recsys_type == RecsysType.TWHIN:
            try:
                latest_post_time = post_table[-1]['created_at']
                second_latest_post_time = post_table[-2]['created_at'] if len(post_table) > 1 else latest_post_time
                post_query = '\n                    SELECT COUNT(*)\n                    FROM post\n                    WHERE created_at = ? OR created_at = ?\n                '
                self.pl_utils._execute_db_command(post_query, (latest_post_time, second_latest_post_time))
                result = self.db_cursor.fetchone()
                latest_post_count = result[0]
                if not latest_post_count:
                    return {'success': False, 'message': 'Fail to get latest posts count'}
                new_rec_matrix = rec_sys_personalized_twh(user_table, post_table, latest_post_count, trace_table, rec_matrix, self.max_rec_post_len, self.sandbox_clock.time_step, use_openai_embedding=self.use_openai_embedding)
            except Exception as e:
                twitter_log.error(e)
                return
        elif self.recsys_type == RecsysType.REDDIT:
            new_rec_matrix = rec_sys_reddit(post_table, rec_matrix, self.max_rec_post_len)
        else:
            raise ValueError('Unsupported recommendation system type, please check the `RecsysType`.')
        sql_query = 'DELETE FROM rec'
        self.pl_utils._execute_db_command(sql_query, commit=True)
        insert_values = [(user_id, post_id) for user_id in range(len(new_rec_matrix)) for post_id in new_rec_matrix[user_id]]
        self.pl_utils._execute_many_db_command('INSERT INTO rec (user_id, post_id) VALUES (?, ?)', insert_values, commit=True)

    async def create_post(self, agent_id: int, content: str):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            post_insert_query = 'INSERT INTO post (user_id, content, created_at, num_likes, num_dislikes, num_shares) VALUES (?, ?, ?, ?, ?, ?)'
            self.pl_utils._execute_db_command(post_insert_query, (user_id, content, current_time, 0, 0, 0), commit=True)
            post_id = self.db_cursor.lastrowid
            action_info = {'content': content, 'post_id': post_id}
            self.pl_utils._record_trace(user_id, ActionType.CREATE_POST.value, action_info, current_time)
            return {'success': True, 'post_id': post_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def repost(self, agent_id: int, post_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            repost_check_query = "SELECT * FROM 'post' WHERE original_post_id = ? AND user_id = ?"
            self.pl_utils._execute_db_command(repost_check_query, (post_id, user_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'Repost record already exists.'}
            post_type_result = self.pl_utils._get_post_type(post_id)
            post_insert_query = 'INSERT INTO post (user_id, original_post_id, created_at) VALUES (?, ?, ?)'
            update_shares_query = 'UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?'
            if not post_type_result:
                return {'success': False, 'error': 'Post not found.'}
            elif post_type_result['type'] == 'common' or post_type_result['type'] == 'quote':
                self.pl_utils._execute_db_command(post_insert_query, (user_id, post_id, current_time), commit=True)
                self.pl_utils._execute_db_command(update_shares_query, (post_id,), commit=True)
            elif post_type_result['type'] == 'repost':
                repost_check_query = "SELECT * FROM 'post' WHERE original_post_id = ? AND user_id = ?"
                self.pl_utils._execute_db_command(repost_check_query, (post_type_result['root_post_id'], user_id))
                if self.db_cursor.fetchone():
                    return {'success': False, 'error': 'Repost record already exists.'}
                self.pl_utils._execute_db_command(post_insert_query, (user_id, post_type_result['root_post_id'], current_time), commit=True)
                self.pl_utils._execute_db_command(update_shares_query, (post_type_result['root_post_id'],), commit=True)
            new_post_id = self.db_cursor.lastrowid
            action_info = {'reposted_id': post_id, 'new_post_id': new_post_id}
            self.pl_utils._record_trace(user_id, ActionType.REPOST.value, action_info, current_time)
            return {'success': True, 'post_id': new_post_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def quote_post(self, agent_id: int, quote_message: tuple):
        post_id, quote_content = quote_message
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            post_query = 'SELECT content FROM post WHERE post_id = ?'
            post_type_result = self.pl_utils._get_post_type(post_id)
            post_insert_query = 'INSERT INTO post (user_id, original_post_id, content, quote_content, created_at) VALUES (?, ?, ?, ?, ?)'
            update_shares_query = 'UPDATE post SET num_shares = num_shares + 1 WHERE post_id = ?'
            if not post_type_result:
                return {'success': False, 'error': 'Post not found.'}
            elif post_type_result['type'] == 'common':
                self.pl_utils._execute_db_command(post_query, (post_id,))
                post_content = self.db_cursor.fetchone()[0]
                self.pl_utils._execute_db_command(post_insert_query, (user_id, post_id, post_content, quote_content, current_time), commit=True)
                self.pl_utils._execute_db_command(update_shares_query, (post_id,), commit=True)
            elif post_type_result['type'] == 'repost' or post_type_result['type'] == 'quote':
                self.pl_utils._execute_db_command(post_query, (post_type_result['root_post_id'],))
                post_content = self.db_cursor.fetchone()[0]
                self.pl_utils._execute_db_command(post_insert_query, (user_id, post_type_result['root_post_id'], post_content, quote_content, current_time), commit=True)
                self.pl_utils._execute_db_command(update_shares_query, (post_type_result['root_post_id'],), commit=True)
            new_post_id = self.db_cursor.lastrowid
            action_info = {'quoted_id': post_id, 'new_post_id': new_post_id}
            self.pl_utils._record_trace(user_id, ActionType.QUOTE_POST.value, action_info, current_time)
            return {'success': True, 'post_id': new_post_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def like_post(self, agent_id: int, post_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id
            like_check_query = "SELECT * FROM 'like' WHERE post_id = ? AND user_id = ?"
            self.pl_utils._execute_db_command(like_check_query, (post_id, user_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'Like record already exists.'}
            if self.allow_self_rating is False:
                check_result = self.pl_utils._check_self_post_rating(post_id, user_id)
                if check_result:
                    return check_result
            post_update_query = 'UPDATE post SET num_likes = num_likes + 1 WHERE post_id = ?'
            self.pl_utils._execute_db_command(post_update_query, (post_id,), commit=True)
            like_insert_query = "INSERT INTO 'like' (post_id, user_id, created_at) VALUES (?, ?, ?)"
            self.pl_utils._execute_db_command(like_insert_query, (post_id, user_id, current_time), commit=True)
            like_id = self.db_cursor.lastrowid
            action_info = {'post_id': post_id, 'like_id': like_id}
            self.pl_utils._record_trace(user_id, ActionType.LIKE_POST.value, action_info, current_time)
            return {'success': True, 'like_id': like_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def unlike_post(self, agent_id: int, post_id: int):
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id
            like_check_query = "SELECT * FROM 'like' WHERE post_id = ? AND user_id = ?"
            self.pl_utils._execute_db_command(like_check_query, (post_id, user_id))
            result = self.db_cursor.fetchone()
            if not result:
                return {'success': False, 'error': 'Like record does not exist.'}
            like_id, _, _, _ = result
            post_update_query = 'UPDATE post SET num_likes = num_likes - 1 WHERE post_id = ?'
            self.pl_utils._execute_db_command(post_update_query, (post_id,), commit=True)
            like_delete_query = "DELETE FROM 'like' WHERE like_id = ?"
            self.pl_utils._execute_db_command(like_delete_query, (like_id,), commit=True)
            action_info = {'post_id': post_id, 'like_id': like_id}
            self.pl_utils._record_trace(user_id, ActionType.UNLIKE_POST.value, action_info)
            return {'success': True, 'like_id': like_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def dislike_post(self, agent_id: int, post_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id
            like_check_query = "SELECT * FROM 'dislike' WHERE post_id = ? AND user_id = ?"
            self.pl_utils._execute_db_command(like_check_query, (post_id, user_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'Dislike record already exists.'}
            if self.allow_self_rating is False:
                check_result = self.pl_utils._check_self_post_rating(post_id, user_id)
                if check_result:
                    return check_result
            post_update_query = 'UPDATE post SET num_dislikes = num_dislikes + 1 WHERE post_id = ?'
            self.pl_utils._execute_db_command(post_update_query, (post_id,), commit=True)
            dislike_insert_query = "INSERT INTO 'dislike' (post_id, user_id, created_at) VALUES (?, ?, ?)"
            self.pl_utils._execute_db_command(dislike_insert_query, (post_id, user_id, current_time), commit=True)
            dislike_id = self.db_cursor.lastrowid
            action_info = {'post_id': post_id, 'dislike_id': dislike_id}
            self.pl_utils._record_trace(user_id, ActionType.DISLIKE_POST.value, action_info, current_time)
            return {'success': True, 'dislike_id': dislike_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def undo_dislike_post(self, agent_id: int, post_id: int):
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id
            like_check_query = "SELECT * FROM 'dislike' WHERE post_id = ? AND user_id = ?"
            self.pl_utils._execute_db_command(like_check_query, (post_id, user_id))
            result = self.db_cursor.fetchone()
            if not result:
                return {'success': False, 'error': 'Dislike record does not exist.'}
            dislike_id, _, _, _ = result
            post_update_query = 'UPDATE post SET num_dislikes = num_dislikes - 1 WHERE post_id = ?'
            self.pl_utils._execute_db_command(post_update_query, (post_id,), commit=True)
            like_delete_query = "DELETE FROM 'dislike' WHERE dislike_id = ?"
            self.pl_utils._execute_db_command(like_delete_query, (dislike_id,), commit=True)
            action_info = {'post_id': post_id, 'dislike_id': dislike_id}
            self.pl_utils._record_trace(user_id, ActionType.UNDO_DISLIKE_POST.value, action_info)
            return {'success': True, 'dislike_id': dislike_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def search_posts(self, agent_id: int, query: str):
        try:
            user_id = agent_id
            sql_query = 'SELECT post_id, user_id, original_post_id, content, quote_content, created_at, num_likes, num_dislikes, num_shares FROM post WHERE content LIKE ? OR CAST(post_id AS TEXT) LIKE ? OR CAST(user_id AS TEXT) LIKE ?'
            self.pl_utils._execute_db_command(sql_query, ('%' + query + '%', '%' + query + '%', '%' + query + '%'), commit=True)
            results = self.db_cursor.fetchall()
            action_info = {'query': query}
            self.pl_utils._record_trace(user_id, ActionType.SEARCH_POSTS.value, action_info)
            if not results:
                return {'success': False, 'message': 'No posts found matching the query.'}
            results_with_comments = self.pl_utils._add_comments_to_posts(results)
            return {'success': True, 'posts': results_with_comments}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def search_user(self, agent_id: int, query: str):
        try:
            user_id = agent_id
            sql_query = 'SELECT user_id, user_name, name, bio, created_at, num_followings, num_followers FROM user WHERE user_name LIKE ? OR name LIKE ? OR bio LIKE ? OR CAST(user_id AS TEXT) LIKE ?'
            self.pl_utils._execute_db_command(sql_query, ('%' + query + '%', '%' + query + '%', '%' + query + '%', '%' + query + '%'), commit=True)
            results = self.db_cursor.fetchall()
            action_info = {'query': query}
            self.pl_utils._record_trace(user_id, ActionType.SEARCH_USER.value, action_info)
            if not results:
                return {'success': False, 'message': 'No users found matching the query.'}
            users = [{'user_id': user_id, 'user_name': user_name, 'name': name, 'bio': bio, 'created_at': created_at, 'num_followings': num_followings, 'num_followers': num_followers} for user_id, user_name, name, bio, created_at, num_followings, num_followers in results]
            return {'success': True, 'users': users}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def follow(self, agent_id: int, followee_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            follow_check_query = 'SELECT * FROM follow WHERE follower_id = ? AND followee_id = ?'
            self.pl_utils._execute_db_command(follow_check_query, (user_id, followee_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'Follow record already exists.'}
            follow_insert_query = 'INSERT INTO follow (follower_id, followee_id, created_at) VALUES (?, ?, ?)'
            self.pl_utils._execute_db_command(follow_insert_query, (user_id, followee_id, current_time), commit=True)
            follow_id = self.db_cursor.lastrowid
            user_update_query1 = 'UPDATE user SET num_followings = num_followings + 1 WHERE user_id = ?'
            self.pl_utils._execute_db_command(user_update_query1, (user_id,), commit=True)
            user_update_query2 = 'UPDATE user SET num_followers = num_followers + 1 WHERE user_id = ?'
            self.pl_utils._execute_db_command(user_update_query2, (followee_id,), commit=True)
            action_info = {'follow_id': follow_id}
            self.pl_utils._record_trace(user_id, ActionType.FOLLOW.value, action_info, current_time)
            return {'success': True, 'follow_id': follow_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def unfollow(self, agent_id: int, followee_id: int):
        try:
            user_id = agent_id
            follow_check_query = 'SELECT follow_id FROM follow WHERE follower_id = ? AND followee_id = ?'
            self.pl_utils._execute_db_command(follow_check_query, (user_id, followee_id))
            follow_record = self.db_cursor.fetchone()
            if not follow_record:
                return {'success': False, 'error': 'Follow record does not exist.'}
            follow_id = follow_record[0]
            follow_delete_query = 'DELETE FROM follow WHERE follow_id = ?'
            self.pl_utils._execute_db_command(follow_delete_query, (follow_id,), commit=True)
            user_update_query1 = 'UPDATE user SET num_followings = num_followings - 1 WHERE user_id = ?'
            self.pl_utils._execute_db_command(user_update_query1, (user_id,), commit=True)
            user_update_query2 = 'UPDATE user SET num_followers = num_followers - 1 WHERE user_id = ?'
            self.pl_utils._execute_db_command(user_update_query2, (followee_id,), commit=True)
            action_info = {'followee_id': followee_id}
            self.pl_utils._record_trace(user_id, ActionType.UNFOLLOW.value, action_info)
            return {'success': True, 'follow_id': follow_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def mute(self, agent_id: int, mutee_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            mute_check_query = 'SELECT * FROM mute WHERE muter_id = ? AND mutee_id = ?'
            self.pl_utils._execute_db_command(mute_check_query, (user_id, mutee_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'Mute record already exists.'}
            mute_insert_query = 'INSERT INTO mute (muter_id, mutee_id, created_at) VALUES (?, ?, ?)'
            self.pl_utils._execute_db_command(mute_insert_query, (user_id, mutee_id, current_time), commit=True)
            mute_id = self.db_cursor.lastrowid
            action_info = {'mutee_id': mutee_id}
            self.pl_utils._record_trace(user_id, ActionType.MUTE.value, action_info, current_time)
            return {'success': True, 'mute_id': mute_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def unmute(self, agent_id: int, mutee_id: int):
        try:
            user_id = agent_id
            mute_check_query = 'SELECT mute_id FROM mute WHERE muter_id = ? AND mutee_id = ?'
            self.pl_utils._execute_db_command(mute_check_query, (user_id, mutee_id))
            mute_record = self.db_cursor.fetchone()
            if not mute_record:
                return {'success': False, 'error': 'No mute record exists.'}
            mute_id = mute_record[0]
            mute_delete_query = 'DELETE FROM mute WHERE mute_id = ?'
            self.pl_utils._execute_db_command(mute_delete_query, (mute_id,), commit=True)
            action_info = {'mutee_id': mutee_id}
            self.pl_utils._record_trace(user_id, ActionType.UNMUTE.value, action_info)
            return {'success': True, 'mute_id': mute_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def trend(self, agent_id: int):
        """
        Get the top K trending posts in the last num_days days.
        """
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            if self.recsys_type == RecsysType.REDDIT:
                start_time = current_time - timedelta(days=self.trend_num_days)
            else:
                start_time = int(current_time) - self.trend_num_days * 24 * 60
            sql_query = '\n                SELECT post_id, user_id, original_post_id, content,\n                quote_content, created_at, num_likes, num_dislikes,\n                num_shares FROM post\n                WHERE created_at >= ?\n                ORDER BY num_likes DESC\n                LIMIT ?\n            '
            self.pl_utils._execute_db_command(sql_query, (start_time, self.trend_top_k), commit=True)
            results = self.db_cursor.fetchall()
            if not results:
                return {'success': False, 'message': 'No trending posts in the specified period.'}
            results_with_comments = self.pl_utils._add_comments_to_posts(results)
            action_info = {'posts': results_with_comments}
            self.pl_utils._record_trace(user_id, ActionType.TREND.value, action_info, current_time)
            return {'success': True, 'posts': results_with_comments}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def create_comment(self, agent_id: int, comment_message: tuple):
        post_id, content = comment_message
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result['type'] == 'repost':
                post_id = post_type_result['root_post_id']
            user_id = agent_id
            comment_insert_query = 'INSERT INTO comment (post_id, user_id, content, created_at) VALUES (?, ?, ?, ?)'
            self.pl_utils._execute_db_command(comment_insert_query, (post_id, user_id, content, current_time), commit=True)
            comment_id = self.db_cursor.lastrowid
            action_info = {'content': content, 'comment_id': comment_id}
            self.pl_utils._record_trace(user_id, ActionType.CREATE_COMMENT.value, action_info, current_time)
            return {'success': True, 'comment_id': comment_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def like_comment(self, agent_id: int, comment_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            like_check_query = 'SELECT * FROM comment_like WHERE comment_id = ? AND user_id = ?'
            self.pl_utils._execute_db_command(like_check_query, (comment_id, user_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'Comment like record already exists.'}
            if self.allow_self_rating is False:
                check_result = self.pl_utils._check_self_comment_rating(comment_id, user_id)
                if check_result:
                    return check_result
            comment_update_query = 'UPDATE comment SET num_likes = num_likes + 1 WHERE comment_id = ?'
            self.pl_utils._execute_db_command(comment_update_query, (comment_id,), commit=True)
            like_insert_query = 'INSERT INTO comment_like (comment_id, user_id, created_at) VALUES (?, ?, ?)'
            self.pl_utils._execute_db_command(like_insert_query, (comment_id, user_id, current_time), commit=True)
            comment_like_id = self.db_cursor.lastrowid
            action_info = {'comment_id': comment_id, 'comment_like_id': comment_like_id}
            self.pl_utils._record_trace(user_id, ActionType.LIKE_COMMENT.value, action_info, current_time)
            return {'success': True, 'comment_like_id': comment_like_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def unlike_comment(self, agent_id: int, comment_id: int):
        try:
            user_id = agent_id
            like_check_query = 'SELECT * FROM comment_like WHERE comment_id = ? AND user_id = ?'
            self.pl_utils._execute_db_command(like_check_query, (comment_id, user_id))
            result = self.db_cursor.fetchone()
            if not result:
                return {'success': False, 'error': 'Comment like record does not exist.'}
            comment_like_id = result[0]
            comment_update_query = 'UPDATE comment SET num_likes = num_likes - 1 WHERE comment_id = ?'
            self.pl_utils._execute_db_command(comment_update_query, (comment_id,), commit=True)
            like_delete_query = 'DELETE FROM comment_like WHERE comment_like_id = ?'
            self.pl_utils._execute_db_command(like_delete_query, (comment_like_id,), commit=True)
            action_info = {'comment_id': comment_id, 'comment_like_id': comment_like_id}
            self.pl_utils._record_trace(user_id, ActionType.UNLIKE_COMMENT.value, action_info)
            return {'success': True, 'comment_like_id': comment_like_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def dislike_comment(self, agent_id: int, comment_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            dislike_check_query = 'SELECT * FROM comment_dislike WHERE comment_id = ? AND user_id = ?'
            self.pl_utils._execute_db_command(dislike_check_query, (comment_id, user_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'Comment dislike record already exists.'}
            if self.allow_self_rating is False:
                check_result = self.pl_utils._check_self_comment_rating(comment_id, user_id)
                if check_result:
                    return check_result
            comment_update_query = 'UPDATE comment SET num_dislikes = num_dislikes + 1 WHERE comment_id = ?'
            self.pl_utils._execute_db_command(comment_update_query, (comment_id,), commit=True)
            dislike_insert_query = 'INSERT INTO comment_dislike (comment_id, user_id, created_at) VALUES (?, ?, ?)'
            self.pl_utils._execute_db_command(dislike_insert_query, (comment_id, user_id, current_time), commit=True)
            comment_dislike_id = self.db_cursor.lastrowid
            action_info = {'comment_id': comment_id, 'comment_dislike_id': comment_dislike_id}
            self.pl_utils._record_trace(user_id, ActionType.DISLIKE_COMMENT.value, action_info, current_time)
            return {'success': True, 'comment_dislike_id': comment_dislike_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def undo_dislike_comment(self, agent_id: int, comment_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            dislike_check_query = 'SELECT comment_dislike_id FROM comment_dislike WHERE comment_id = ? AND user_id = ?'
            self.pl_utils._execute_db_command(dislike_check_query, (comment_id, user_id))
            dislike_record = self.db_cursor.fetchone()
            if not dislike_record:
                return {'success': False, 'error': 'Comment dislike record does not exist.'}
            comment_dislike_id = dislike_record[0]
            dislike_delete_query = 'DELETE FROM comment_dislike WHERE comment_id = ? AND user_id = ?'
            self.pl_utils._execute_db_command(dislike_delete_query, (comment_id, user_id), commit=True)
            comment_update_query = 'UPDATE comment SET num_dislikes = num_dislikes - 1 WHERE comment_id = ?'
            self.pl_utils._execute_db_command(comment_update_query, (comment_id,), commit=True)
            action_info = {'comment_id': comment_id, 'comment_dislike_id': comment_dislike_id}
            self.pl_utils._record_trace(user_id, ActionType.UNDO_DISLIKE_COMMENT.value, action_info, current_time)
            return {'success': True, 'comment_dislike_id': comment_dislike_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def do_nothing(self, agent_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            action_info = {}
            self.pl_utils._record_trace(user_id, ActionType.DO_NOTHING.value, action_info, current_time)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def interview(self, agent_id: int, interview_data):
        """Interview an agent with the given prompt and record the response.

        Args:
            agent_id (int): The ID of the agent being interviewed.
            interview_data: Either a string (prompt only) or dict with prompt
                and response.

        Returns:
            dict: A dictionary with success status.
        """
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            if isinstance(interview_data, str):
                prompt = interview_data
                response = None
                interview_id = f'{current_time}_{user_id}'
                action_info = {'prompt': prompt, 'interview_id': interview_id}
            else:
                prompt = interview_data.get('prompt', '')
                response = interview_data.get('response', '')
                interview_id = f'{current_time}_{user_id}'
                action_info = {'prompt': prompt, 'response': response, 'interview_id': interview_id}
            self.pl_utils._record_trace(user_id, ActionType.INTERVIEW.value, action_info, current_time)
            return {'success': True, 'interview_id': interview_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def report_post(self, agent_id: int, report_message: tuple):
        post_id, report_reason = report_message
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            post_type_result = self.pl_utils._get_post_type(post_id)
            check_report_query = 'SELECT * FROM report WHERE user_id = ? AND post_id = ?'
            self.pl_utils._execute_db_command(check_report_query, (user_id, post_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'Report record already exists.'}
            if not post_type_result:
                return {'success': False, 'error': 'Post not found.'}
            update_reports_query = 'UPDATE post SET num_reports = num_reports + 1 WHERE post_id = ?'
            self.pl_utils._execute_db_command(update_reports_query, (post_id,), commit=True)
            report_insert_query = 'INSERT INTO report (post_id, user_id, report_reason, created_at) VALUES (?, ?, ?, ?)'
            self.pl_utils._execute_db_command(report_insert_query, (post_id, user_id, report_reason, current_time), commit=True)
            report_id = self.db_cursor.lastrowid
            action_info = {'post_id': post_id, 'report_id': report_id}
            self.pl_utils._record_trace(user_id, ActionType.REPORT_POST.value, action_info, current_time)
            return {'success': True, 'report_id': report_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def send_to_group(self, agent_id: int, message: tuple):
        group_id, content = message
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            check_query = 'SELECT * FROM group_members WHERE group_id = ? AND agent_id = ?'
            self.pl_utils._execute_db_command(check_query, (group_id, user_id))
            if not self.db_cursor.fetchone():
                return {'success': False, 'error': 'User is not a member of this group.'}
            insert_query = '\n                INSERT INTO group_messages\n                (group_id, sender_id, content, sent_at)\n                VALUES (?, ?, ?, ?)\n            '
            self.pl_utils._execute_db_command(insert_query, (group_id, user_id, content, current_time), commit=True)
            message_id = self.db_cursor.lastrowid
            members_query = 'SELECT agent_id FROM group_members WHERE group_id = ? AND agent_id != ?'
            self.pl_utils._execute_db_command(members_query, (group_id, user_id))
            members = [row[0] for row in self.db_cursor.fetchall()]
            action_info = {'group_id': group_id, 'message_id': message_id, 'content': content}
            self.pl_utils._record_trace(user_id, ActionType.SEND_TO_GROUP.value, action_info, current_time)
            return {'success': True, 'message_id': message_id, 'to': members}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def create_group(self, agent_id: int, group_name: str):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            insert_query = '\n                INSERT INTO chat_group (name, created_at) VALUES (?, ?)\n            '
            self.pl_utils._execute_db_command(insert_query, (group_name, current_time), commit=True)
            group_id = self.db_cursor.lastrowid
            join_query = '\n                INSERT INTO group_members (group_id, agent_id, joined_at)\n                VALUES (?, ?, ?)\n            '
            self.pl_utils._execute_db_command(join_query, (group_id, user_id, current_time), commit=True)
            action_info = {'group_id': group_id, 'group_name': group_name}
            self.pl_utils._record_trace(user_id, ActionType.CREATE_GROUP.value, action_info, current_time)
            return {'success': True, 'group_id': group_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def join_group(self, agent_id: int, group_id: int):
        if self.recsys_type == RecsysType.REDDIT:
            current_time = self.sandbox_clock.time_transfer(datetime.now(), self.start_time)
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            user_id = agent_id
            check_group_query = 'SELECT * FROM chat_group\n                WHERE group_id = ?'
            self.pl_utils._execute_db_command(check_group_query, (group_id,))
            if not self.db_cursor.fetchone():
                return {'success': False, 'error': 'Group does not exist.'}
            check_member_query = 'SELECT * FROM group_members WHERE group_id = ? AND agent_id = ?'
            self.pl_utils._execute_db_command(check_member_query, (group_id, user_id))
            if self.db_cursor.fetchone():
                return {'success': False, 'error': 'User is already in the group.'}
            join_query = '\n                INSERT INTO group_members\n                (group_id, agent_id, joined_at) VALUES (?, ?, ?)\n            '
            self.pl_utils._execute_db_command(join_query, (group_id, user_id, current_time), commit=True)
            action_info = {'group_id': group_id}
            self.pl_utils._record_trace(user_id, ActionType.JOIN_GROUP.value, action_info, current_time)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def leave_group(self, agent_id: int, group_id: int):
        try:
            user_id = agent_id
            check_query = 'SELECT * FROM group_members WHERE group_id = ? AND agent_id = ?'
            self.pl_utils._execute_db_command(check_query, (group_id, user_id))
            if not self.db_cursor.fetchone():
                return {'success': False, 'error': 'User is not a member of this group.'}
            delete_query = 'DELETE FROM group_members WHERE group_id = ? AND agent_id = ?'
            self.pl_utils._execute_db_command(delete_query, (group_id, user_id), commit=True)
            action_info = {'group_id': group_id}
            self.pl_utils._record_trace(user_id, ActionType.LEAVE_GROUP.value, action_info)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def listen_from_group(self, agent_id: int):
        try:
            query = ' SELECT * FROM chat_group '
            self.pl_utils._execute_db_command(query)
            all_groups = {}
            for row in self.db_cursor.fetchall():
                all_groups[row[0]] = row[1]
            in_query = '\n                SELECT group_id FROM group_members WHERE agent_id = ?\n            '
            self.pl_utils._execute_db_command(in_query, (agent_id,))
            joined_group_ids = [row[0] for row in self.db_cursor.fetchall()]
            messages = {}
            for group_id in joined_group_ids:
                select_query = '\n                    SELECT message_id, content, sender_id,\n                    sent_at FROM group_messages WHERE group_id = ?\n                '
                self.pl_utils._execute_db_command(select_query, (group_id,))
                messages[group_id] = [{'message_id': row[0], 'content': row[1], 'sender_id': row[2], 'sent_at': row[3]} for row in self.db_cursor.fetchall()]
            return {'success': True, 'all_groups': all_groups, 'joined_groups': joined_group_ids, 'messages': messages}
        except Exception as e:
            return {'success': False, 'error': str(e)}


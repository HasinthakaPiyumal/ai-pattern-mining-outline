# Cluster 24

def print_db_contents(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_log.info('Tables:' + ' '.join([str(table[0]) for table in tables]))
    for table_name in tables:
        table_log.info(f'\nTable: {table_name[0]}')
        cursor.execute(f'PRAGMA table_info({table_name[0]})')
        columns = cursor.fetchall()
        table_log.info('Columns:')
        for col in columns:
            table_log.info(f'  {col[1]} ({col[2]})')
        cursor.execute(f'SELECT * FROM {table_name[0]}')
        rows = cursor.fetchall()
        table_log.info('Contents:')
        for row in rows:
            table_log.info(' ' + ', '.join((str(item) for item in row)))
    conn.close()

def generate_user_actions(n_users, posts_per_user):
    """
    Generate a list of user actions for n users with different posting
    behaviors. 1/3 of the users each sending m posts, 1/3 sending 1 post,
    and 1/3 not posting at all.
    """
    actions = []
    users_per_group = n_users // 3
    for user_id in range(1, n_users + 1):
        user_message = ('username' + str(user_id), 'name' + str(user_id), 'No descrption.')
        actions.append((user_id, user_message, 'sign_up'))
        if user_id <= users_per_group:
            for post_num in range(1, posts_per_user + 1):
                actions.append((user_id, f'This is post {post_num} from User{user_id}', 'create_post'))
        elif user_id <= 2 * users_per_group:
            actions.append((user_id, f'This is post 1 from User{user_id}', 'create_post'))
    return actions


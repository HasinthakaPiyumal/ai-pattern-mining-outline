# Cluster 12

def main(sqlite_db_path):
    neo4j_driver = connect_to_neo4j(neo4j_config)
    sqlite_conn = connect_to_sqlite(sqlite_db_path)
    sqlite_cursor = sqlite_conn.cursor()
    with neo4j_driver.session() as session:
        sqlite_cursor.execute('SELECT user_id, user_name, name, bio, created_at FROM user ORDER BY created_at')
        for row in sqlite_cursor:
            user_id, user_name, name, bio, created_at = row
            info_dict = {'user_name': user_name, 'name': name, 'bio': bio}
            print('info_dict:\n', info_dict)
            session.execute_write(create_user_node, user_id, info_dict, created_at)
        sqlite_cursor.execute('SELECT follower_id, followee_id, created_at FROM follow ORDER BY created_at')
        for row in sqlite_cursor:
            follower_id, followee_id, created_at = row
            print(f'follower_id:{follower_id}, followee_id:{followee_id}, created_at:{created_at}')
            session.execute_write(create_follow_relationship, follower_id, followee_id, created_at)
    sqlite_conn.close()
    neo4j_driver.close()

def connect_to_neo4j(config):
    return GraphDatabase.driver(config.uri, auth=(config.username, config.password))

def connect_to_sqlite(db_path):
    return sqlite3.connect(db_path)

def main(sqlite_db_path):
    neo4j_driver = connect_to_neo4j(neo4j_config)
    sqlite_conn = connect_to_sqlite(sqlite_db_path)
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute('SELECT user_id, created_at, action, info FROM trace ORDER BY created_at')
    with neo4j_driver.session() as session:
        for row in sqlite_cursor:
            user_id, created_at, action, info = row
            info_dict = json.loads(info)
            if action == 'sign_up':
                session.execute_write(create_user_node, user_id, info_dict, created_at)
            elif action == 'follow':
                follow_id = int(info_dict['follow_id'])
                timestamp = int(info_dict['time_stamp'])
                session.execute_write(create_follow_relationship, user_id, follow_id, created_at, timestamp)
    sqlite_conn.close()
    neo4j_driver.close()


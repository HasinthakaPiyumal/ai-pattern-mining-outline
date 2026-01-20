# Cluster 13

def create_user_node(tx, user_id, action_info, created_at):
    formatted_datetime = format_datetime(created_at)
    query = 'MERGE (u:User {user_id: $user_id}) SET u += $action_info, u.created_at = datetime($created_at)'
    tx.run(query, user_id=user_id, action_info=action_info, created_at=formatted_datetime)

def format_datetime(dt_string):
    try:
        dt = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S.%f')
        print(dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z')
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    except Exception:
        return int(dt_string)

def create_follow_relationship(tx, user_id, follow_id, created_at, timestamp):
    formatted_datetime = format_datetime(created_at)
    query = 'MERGE (u1:User {user_id: $user_id}) MERGE (u2:User {user_id: $follow_id}) CREATE (u1)-[:FOLLOWS {created_at: datetime($created_at), timestamp: $timestamp}]->(u2)'
    tx.run(query, user_id=user_id, follow_id=follow_id, created_at=formatted_datetime, timestamp=timestamp)


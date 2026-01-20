# Cluster 52

def _generate_mapping_keys(db_generation_parameters: DBGenerationParameters) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate all the FK mappings between the generated tables based on the provided parameters.
    :param db_generation_parameters: The generation parameters to use.
    :return: Dicts containing the FK information for each table.
    """
    token_dicts: Dict[str, List[Dict[str, Any]]] = {'lidar_pc': [], 'lidar': [], 'image': [], 'camera': [], 'ego_pose': [], 'track': [], 'lidar_box': [], 'scene': [], 'traffic_light_status': [], 'category': [], 'scenario_tag': []}
    num_pc_tokens_per_scene = int(db_generation_parameters.num_sensor_data_per_sensor / db_generation_parameters.num_scenes)
    base_token_multiplier = 100000
    offset_scene_token = 1 * base_token_multiplier
    offset_ego_pose_token = 3 * base_token_multiplier
    offset_traffic_light_status_token = 4 * base_token_multiplier
    offset_lidar_box_token = 5 * base_token_multiplier
    offset_track_token_token = 6 * base_token_multiplier
    offset_lidar_token = 7 * base_token_multiplier
    offset_scenario_tag_token = 8 * base_token_multiplier
    offset_agents_token = 9 * base_token_multiplier
    offset_static_objects_token = 9 * base_token_multiplier + 5
    offset_camera_token = 10 * base_token_multiplier
    offset_image_token = 11 * base_token_multiplier
    step_traffic_light_status_pc = 10000
    step_lidar_box_pc = 10000
    _generate_camera_data(db_generation_parameters, token_dicts, offset_image_token, offset_camera_token, offset_ego_pose_token)
    for sensor_idx in range(db_generation_parameters.num_lidars):
        lidar_token = offset_lidar_token + sensor_idx
        lidar_channel_name = 'channel' if sensor_idx < 1 else f'channel_{sensor_idx}'
        for sensor_data_idx in range(db_generation_parameters.num_sensor_data_per_sensor):
            lidar_pc_token = sensor_data_idx + sensor_idx * db_generation_parameters.num_sensor_data_per_sensor
            scene_token = int(sensor_data_idx / num_pc_tokens_per_scene) + offset_scene_token
            ego_pose_token = sensor_data_idx + offset_ego_pose_token
            timestamp_ego_pose = sensor_data_idx * 1000000.0
            timestamp_lidar_pc = timestamp_ego_pose + sensor_idx
            next_lidar_pc = {'lidar_pc_token': lidar_pc_token, 'prev_lidar_pc_token': None if sensor_data_idx == 0 else sensor_data_idx - 1, 'next_lidar_pc_token': None if sensor_data_idx == db_generation_parameters.num_sensor_data_per_sensor - 1 else sensor_data_idx + 1, 'scene_token': scene_token, 'ego_pose_token': ego_pose_token, 'lidar_token': lidar_token, 'lidar_pc_timestamp': timestamp_lidar_pc}
            token_dicts['lidar_pc'].append(next_lidar_pc)
            ego_pose_entry = {'token': ego_pose_token, 'timestamp': timestamp_ego_pose + 333}
            if ego_pose_entry not in token_dicts['ego_pose']:
                token_dicts['ego_pose'].append(ego_pose_entry)
            lidar_token_entry = {'token': lidar_token, 'channel': lidar_channel_name}
            if lidar_token_entry not in token_dicts['lidar']:
                token_dicts['lidar'].append(lidar_token_entry)
            for traffic_light_idx in range(db_generation_parameters.num_traffic_lights_per_lidar_pc):
                statuses = ['green', 'red', 'yellow', 'unknown']
                traffic_light_status_entry = {'token': offset_traffic_light_status_token + sensor_data_idx * step_traffic_light_status_pc + traffic_light_idx, 'lidar_pc_token': sensor_data_idx, 'lane_connector_id': traffic_light_idx, 'status': statuses[(offset_traffic_light_status_token + traffic_light_idx) % len(statuses)]}
                if traffic_light_status_entry not in token_dicts['traffic_light_status']:
                    token_dicts['traffic_light_status'].append(traffic_light_status_entry)
            for object_idx in range(db_generation_parameters.total_object_count()):
                lidar_box_entry = {'token': offset_lidar_box_token + sensor_data_idx * step_lidar_box_pc + object_idx, 'lidar_pc_token': sensor_data_idx, 'track_token': offset_track_token_token + object_idx, 'prev_token': None if sensor_data_idx == 0 else offset_lidar_box_token + (sensor_data_idx - 1) * step_lidar_box_pc + object_idx, 'next_token': None if sensor_data_idx == db_generation_parameters.num_sensor_data_per_sensor - 1 else offset_lidar_box_token + (sensor_data_idx + 1) * step_lidar_box_pc + object_idx}
                if lidar_box_entry not in token_dicts['lidar_box']:
                    token_dicts['lidar_box'].append(lidar_box_entry)
            if sensor_data_idx % num_pc_tokens_per_scene == 0:
                scene_token_entry = {'token': scene_token, 'ego_pose_token': ego_pose_token + num_pc_tokens_per_scene - 1, 'name': 'scene-{:03d}'.format(int(sensor_data_idx / num_pc_tokens_per_scene))}
                if scene_token_entry not in token_dicts['scene']:
                    token_dicts['scene'].append(scene_token_entry)
                scene_idx = sensor_data_idx // num_pc_tokens_per_scene
                if scene_idx in db_generation_parameters.scene_scenario_tag_mapping and lidar_channel_name == 'channel':
                    tags = db_generation_parameters.scene_scenario_tag_mapping[scene_idx]
                    for tag in tags:
                        row = {'token': offset_scenario_tag_token + len(token_dicts['scenario_tag']), 'lidar_pc_token': lidar_pc_token, 'type': tag}
                        token_dicts['scenario_tag'].append(row)
    for lidar_pc_idx in range(db_generation_parameters.total_object_count()):
        token_dicts['track'].append({'token': offset_track_token_token + lidar_pc_idx, 'category_token': offset_agents_token if lidar_pc_idx < db_generation_parameters.num_agents_per_lidar_pc else offset_static_objects_token})
    return token_dicts

def _generate_camera_data(db_generation_parameters: DBGenerationParameters, token_dicts: Dict[str, List[Dict[str, Any]]], offset_image_token: int, offset_camera_token: int, offset_ego_pose_token: int) -> None:
    """
    Generate all the mappings for the camera table based on the provided parameters.
    :param db_generation_parameters: The generation parameters to use.
    :param token_dicts: Dicts containing the FK information for each table
    :param offset_image_token: Offset to mark the range of the image tokens
    :param offset_camera_token: Offset to mark the range of the camera tokens
    :param offset_ego_pose_token: Offset to mark the range of the ego pose tokens
    """
    for sensor_idx in range(db_generation_parameters.num_cameras):
        camera_token = offset_camera_token + sensor_idx
        camera_channel_name = f'camera_{sensor_idx}'
        for sensor_data_idx in range(db_generation_parameters.num_sensor_data_per_sensor):
            if sensor_data_idx % db_generation_parameters.num_lidarpc_per_image_ratio == 0:
                ego_pose_token = sensor_data_idx + offset_ego_pose_token
                image_timestamp = sensor_data_idx * 1000000.0
                image_token = sensor_data_idx + sensor_idx * db_generation_parameters.num_sensor_data_per_sensor + offset_image_token
                image_entry = {'image_token': image_token, 'prev_image_token': None if sensor_data_idx == 0 else token_dicts['image'][-1]['image_token'], 'next_image_token': image_token + db_generation_parameters.num_lidarpc_per_image_ratio, 'ego_pose_token': ego_pose_token, 'camera_token': camera_token, 'image_timestamp': image_timestamp + 10 + sensor_idx}
                token_dicts['image'].append(image_entry)
        token_dicts['image'][-1]['next_image_token'] = None
        camera_entry = {'token': camera_token, 'channel': camera_channel_name}
        token_dicts['camera'].append(camera_entry)

def _generate_lidar_pc_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the lidar_pc table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data provided to _generate_lidar_pc_table'
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row['lidar_pc_token']), _int_to_token(row['next_lidar_pc_token']), _int_to_token(row['prev_lidar_pc_token']), _int_to_token(row['ego_pose_token']), _int_to_token(row['lidar_token']), _int_to_token(row['scene_token']), f'pc_{idx}.dat', row['lidar_pc_timestamp']))
    query = '\n    CREATE TABLE lidar_pc (\n        token BLOB NOT NULL,\n        next_token BLOB,\n        prev_token BLOB,\n        ego_pose_token BLOB NOT NULL,\n        lidar_token BLOB NOT NULL,\n        scene_token BLOB,\n        filename VARCHAR(128),\n        timestamp INTEGER,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO lidar_pc (\n        token,\n        next_token,\n        prev_token,\n        ego_pose_token,\n        lidar_token,\n        scene_token,\n        filename,\n        timestamp\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def _int_to_token(val: Optional[int]) -> Optional[bytearray]:
    """
    Convert an int directly to a token bytearray.
    Intended for use only in this file.
    :param val: The int to convert.
    :return: The token bytearray.
    """
    return None if val is None else bytearray.fromhex('{:08d}'.format(val))

def _execute_non_query(query_text: str, file_path: str) -> None:
    """
    Connect to a SQLite DB and runs a query that returns no results.
    E.g. a CREATE TABLE statement.
    :param query_text: The query text to run.
    :param file_path: The file on which to run the query.
    """
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()
    try:
        cursor.execute(query_text)
    finally:
        cursor.close()
        connection.close()

def _execute_bulk_insert(query_text: str, values: List[Any], file_path: str) -> None:
    """
    Connect to a SQLite DB and runs a query that inserts many rows into the DB.
    This function will commit the changes after a successful execution.
    :param query_text: The query text to run.
    :param values: The values to insert.
    :param file_path: The file on which to run the query.
    """
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()
    try:
        cursor.executemany(query_text, values)
        cursor.execute('commit;')
    finally:
        cursor.close()
        connection.close()

def _generate_lidar_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the lidar table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data provided to _generate_lidar_table'
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row['token']), _int_to_token(0), row['channel'], 'model', pickle.dumps(Translation([idx, idx + 1, idx + 2])), pickle.dumps(Rotation([0, 0, 0, 0]))))
    query = '\n    CREATE TABLE lidar (\n        token BLOB NOT NULL,\n        log_token BLOB NOT NULL,\n        channel VARCHAR(64),\n        model VARCHAR(64),\n        translation BLOB,\n        rotation BLOB,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO lidar (\n        token,\n        log_token,\n        channel,\n        model,\n        translation,\n        rotation\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_ego_pose_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the ego_pose table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data passed to _generate_ego_pose_table'
    rows = []
    for idx, mapping_dict in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(mapping_dict['token']), mapping_dict['timestamp'], idx, idx + 1, idx + 2, idx + 3, idx + 4, idx + 5, idx + 6, idx + 7, idx + 8, idx + 9, idx + 10, idx + 11, idx + 12, idx + 13, idx + 14, idx + 15, idx + 16, _int_to_token(0)))
    query = '\n    CREATE TABLE ego_pose (\n        token BLOB NOT NULL,\n        timestamp INTEGER,\n        x FLOAT,\n        y FLOAT,\n        z FLOAT,\n        qw FLOAT,\n        qx FLOAT,\n        qy FLOAT,\n        qz FLOAT,\n        vx FLOAT,\n        vy FLOAT,\n        vz FLOAT,\n        acceleration_x FLOAT,\n        acceleration_y FLOAT,\n        acceleration_z FLOAT,\n        angular_rate_x FLOAT,\n        angular_rate_y FLOAT,\n        angular_rate_z FLOAT,\n        epsg INTEGER,\n        log_token BLOB NOT NULL,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO ego_pose (\n        token,\n        timestamp,\n        x,\n        y,\n        z,\n        qw,\n        qx,\n        qy,\n        qz,\n        vx,\n        vy,\n        vz,\n        acceleration_x,\n        acceleration_y,\n        acceleration_z,\n        angular_rate_x,\n        angular_rate_y,\n        angular_rate_z,\n        epsg,\n        log_token\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_scene_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the scene table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data passed to _generate_scene_table'
    rows = []
    for idx, mapping_dict in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(mapping_dict['token']), _int_to_token(0), mapping_dict['name'], _int_to_token(mapping_dict['ego_pose_token']), f'{idx} {idx + 1} {idx + 2}'))
    query = '\n    CREATE TABLE scene (\n        token BLOB NOT NULL,\n        log_token BLOB NOT NULL,\n        name TEXT,\n        goal_ego_pose_token BLOB,\n        roadblock_ids TEXT,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO scene (\n        token,\n        log_token,\n        name,\n        goal_ego_pose_token,\n        roadblock_ids\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_traffic_light_status_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generates the traffic_light_status table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data passed to _generate_traffic_light_status_table'
    rows = []
    for idx, mapping_dict in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(mapping_dict['token']), _int_to_token(mapping_dict['lidar_pc_token']), mapping_dict['lane_connector_id'], mapping_dict['status']))
    query = '\n    CREATE TABLE traffic_light_status (\n        token BLOB NOT NULL,\n        lidar_pc_token BLOB NOT NULL,\n        lane_connector_id INTEGER,\n        status VARCHAR(8),\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO traffic_light_status (\n        token,\n        lidar_pc_token,\n        lane_connector_id,\n        status\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_lidar_box_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the lidar_box table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data passed to _generate_lidar_box_table'
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row['token']), _int_to_token(row['lidar_pc_token']), _int_to_token(row['track_token']), _int_to_token(row['next_token']), _int_to_token(row['prev_token']), idx, idx + 1, idx + 2, idx + 3, idx + 4, idx + 5, idx + 6, idx + 7, idx + 8, idx + 9, idx + 10))
    query = '\n    CREATE TABLE lidar_box (\n        token BLOB NOT NULL,\n        lidar_pc_token BLOB NOT NULL,\n        track_token BLOB NOT NULL,\n        next_token BLOB,\n        prev_token BLOB,\n        x FLOAT,\n        y FLOAT,\n        z FLOAT,\n        width FLOAT,\n        length FLOAT,\n        height FLOAT,\n        vx FLOAT,\n        vy FLOAT,\n        vz FLOAT,\n        yaw FLOAT,\n        confidence FLOAT,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO lidar_box (\n        token,\n        lidar_pc_token,\n        track_token,\n        next_token,\n        prev_token,\n        x,\n        y,\n        z,\n        width,\n        length,\n        height,\n        vx,\n        vy,\n        vz,\n        yaw,\n        confidence\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_category_table(file_path: str) -> None:
    """
    Generate the category table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    categories = ['vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier', 'czone_sign', 'generic_object']
    rows = [(_int_to_token(idx + 900000), cat, cat + '.') for idx, cat in enumerate(categories)]
    query = '\n    CREATE TABLE category (\n        token BLOB NOT NULL,\n        name VARCHAR(64),\n        description TEXT,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO category (\n        token,\n        name,\n        description\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_scenario_tag_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the scenario_tag table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data passed to _generate_scenario_tag_table'
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row['token']), _int_to_token(row['lidar_pc_token']), row['type'], _int_to_token(0)))
    query = '\n        CREATE TABLE scenario_tag (\n            token BLOB NOT NULL,\n            lidar_pc_token BLOB NOT NULL,\n            type TEXT,\n            agent_track_token BLOB,\n            PRIMARY KEY (token)\n        );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO scenario_tag (\n        token,\n        lidar_pc_token,\n        type,\n        agent_track_token\n    )\n    VALUES ({('?,' * len(rows[0]))[:-1]})\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_track_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the track table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data passed to _generate_track_table'
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row['token']), _int_to_token(row['category_token']), idx, idx + 1, idx + 2))
    query = '\n    CREATE TABLE track (\n        token BLOB NOT NULL,\n        category_token BLOB NOT NULL,\n        width FLOAT,\n        length FLOAT,\n        height FLOAT,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO track (\n        token,\n        category_token,\n        width,\n        length,\n        height\n    )\n    VALUES ({('?,' * len(rows[0]))[:-1]})\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_log_table(file_path: str) -> None:
    """
    Generates the log table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    rows = [(_int_to_token(0), 'vehicle_name', 'today', 0, 'logfile', 'location', 'map_version')]
    query = '\n    CREATE TABLE log (\n        token BLOB NOT NULL,\n        vehicle_name VARCHAR(64),\n        date VARCHAR(64),\n        timestamp INTEGER,\n        logfile VARCHAR(64),\n        location VARCHAR(64),\n        map_version VARCHAR(64),\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO log (\n        token,\n        vehicle_name,\n        date,\n        timestamp,\n        logfile,\n        location,\n        map_version\n    )\n    VALUES ({('?,' * len(rows[0]))[:-1]})\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_image_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the image table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data provided to _generate_image_table'
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row['image_token']), _int_to_token(row['next_image_token']), _int_to_token(row['prev_image_token']), _int_to_token(row['ego_pose_token']), _int_to_token(row['camera_token']), f'image_{idx}_channel_{row['camera_token']}.dat', row['image_timestamp']))
    query = '\n    CREATE TABLE image (\n        token BLOB NOT NULL,\n        next_token BLOB,\n        prev_token BLOB,\n        ego_pose_token BLOB NOT NULL,\n        camera_token BLOB NOT NULL,\n        filename_jpg VARCHAR(128),\n        timestamp INTEGER,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO image (\n        token,\n        next_token,\n        prev_token,\n        ego_pose_token,\n        camera_token,\n        filename_jpg,\n        timestamp\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def _generate_camera_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the camera table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, 'No data provided to _generate_lidar_table'
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row['token']), _int_to_token(0), row['channel'], 'model', pickle.dumps(Translation([idx, idx + 1, idx + 2])), pickle.dumps(Rotation([0, 0, 0, 0])), pickle.dumps(CameraIntrinsic([[0, 0, 0], [0, 0, 0], [0, 0, 0]])), pickle.dumps([0, 1, 2, 3, 4]), 1.23, 3.21))
    query = '\n    CREATE TABLE camera (\n        token BLOB NOT NULL,\n        log_token BLOB NOT NULL,\n        channel VARCHAR(64),\n        model VARCHAR(64),\n        translation BLOB,\n        rotation BLOB,\n        intrinsic BLOB,\n        distortion BLOB,\n        width FLOAT,\n        height FLOAT,\n        PRIMARY KEY (token)\n    );\n    '
    _execute_non_query(query, file_path)
    query = f'\n    INSERT INTO camera (\n        token,\n        log_token,\n        channel,\n        model,\n        translation,\n        rotation,\n        intrinsic,\n        distortion,\n        width,\n        height\n    )\n    VALUES({('?,' * len(rows[0]))[:-1]});\n    '
    _execute_bulk_insert(query, rows, file_path)

def generate_minimal_nuplan_db(parameters: DBGenerationParameters) -> None:
    """
    Generate a synthetic nuplan_db based on the supplied generation parameters.
    :param parameters: The parameters to use for generation.
    """
    mapping_keys = _generate_mapping_keys(parameters)
    _generate_lidar_pc_table(mapping_keys['lidar_pc'], parameters.file_path)
    _generate_lidar_table(mapping_keys['lidar'], parameters.file_path)
    _generate_image_table(mapping_keys['image'], parameters.file_path)
    _generate_camera_table(mapping_keys['camera'], parameters.file_path)
    _generate_ego_pose_table(mapping_keys['ego_pose'], parameters.file_path)
    _generate_scene_table(mapping_keys['scene'], parameters.file_path)
    _generate_traffic_light_status_table(mapping_keys['traffic_light_status'], parameters.file_path)
    _generate_lidar_box_table(mapping_keys['lidar_box'], parameters.file_path)
    _generate_track_table(mapping_keys['track'], parameters.file_path)
    _generate_scenario_tag_table(mapping_keys['scenario_tag'], parameters.file_path)
    _generate_category_table(parameters.file_path)
    _generate_log_table(parameters.file_path)

class TestDbCliQueries(unittest.TestCase):
    """
    Test suite for the DB Cli queries.
    """

    @staticmethod
    def getDBFilePath() -> Path:
        """
        Get the location for the temporary SQLite file used for the test DB.
        :return: The filepath for the test data.
        """
        return Path('/tmp/test_db_cli_queries.sqlite3')

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create the mock DB data.
        """
        db_file_path = TestDbCliQueries.getDBFilePath()
        if db_file_path.exists():
            db_file_path.unlink()
        generation_parameters = DBGenerationParameters(num_lidars=1, num_cameras=2, num_sensor_data_per_sensor=50, num_lidarpc_per_image_ratio=2, num_scenes=10, num_traffic_lights_per_lidar_pc=5, num_agents_per_lidar_pc=3, num_static_objects_per_lidar_pc=2, scene_scenario_tag_mapping={5: ['first_tag'], 6: ['first_tag', 'second_tag']}, file_path=str(db_file_path))
        generate_minimal_nuplan_db(generation_parameters)

    def setUp(self) -> None:
        """
        The method to run before each test.
        """
        self.db_file_name = str(TestDbCliQueries.getDBFilePath())

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Destroy the mock DB data.
        """
        db_file_path = TestDbCliQueries.getDBFilePath()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)

    def test_get_db_description(self) -> None:
        """
        Test the get_db_description queries.
        """
        db_description = get_db_description(self.db_file_name)
        expected_tables = ['category', 'ego_pose', 'lidar', 'lidar_box', 'lidar_pc', 'log', 'scenario_tag', 'scene', 'track', 'traffic_light_status', 'camera', 'image']
        self.assertEqual(len(expected_tables), len(db_description.tables))
        for expected_table in expected_tables:
            self.assertTrue(expected_table in db_description.tables)
        lidar_pc_table = db_description.tables['lidar_pc']
        self.assertEqual('lidar_pc', lidar_pc_table.name)
        self.assertEqual(50, lidar_pc_table.row_count)
        self.assertEqual(8, len(lidar_pc_table.columns))
        columns = sorted(lidar_pc_table.columns.values(), key=lambda x: x.column_id)

        def _validate_column(column: ColumnDescription, expected_id: int, expected_name: str, expected_data_type: str, expected_nullable: bool, expected_is_primary_key: bool) -> None:
            """
            A quick method to validate column info to reduce boilerplate.
            """
            self.assertEqual(expected_id, column.column_id)
            self.assertEqual(expected_name, column.name)
            self.assertEqual(expected_data_type, column.data_type)
            self.assertEqual(expected_nullable, column.nullable)
            self.assertEqual(expected_is_primary_key, column.is_primary_key)
        _validate_column(columns[0], 0, 'token', 'BLOB', False, True)
        _validate_column(columns[1], 1, 'next_token', 'BLOB', True, False)
        _validate_column(columns[2], 2, 'prev_token', 'BLOB', True, False)
        _validate_column(columns[3], 3, 'ego_pose_token', 'BLOB', False, False)
        _validate_column(columns[4], 4, 'lidar_token', 'BLOB', False, False)
        _validate_column(columns[5], 5, 'scene_token', 'BLOB', True, False)
        _validate_column(columns[6], 6, 'filename', 'VARCHAR(128)', True, False)
        _validate_column(columns[7], 7, 'timestamp', 'INTEGER', True, False)

    def test_get_db_duration_in_us(self) -> None:
        """
        Test the get_db_duration_in_us query
        """
        duration = get_db_duration_in_us(self.db_file_name)
        self.assertEqual(49 * 1000000.0, duration)

    def test_get_db_log_duration(self) -> None:
        """
        Test the get_db_log_duration query.
        """
        log_durations = list(get_db_log_duration(self.db_file_name))
        self.assertEqual(1, len(log_durations))
        self.assertEqual('logfile', log_durations[0][0])
        self.assertEqual(49 * 1000000.0, log_durations[0][1])

    def test_get_db_log_vehicles(self) -> None:
        """
        Test the get_db_log_vehicles query.
        """
        log_vehicles = list(get_db_log_vehicles(self.db_file_name))
        self.assertEqual(1, len(log_vehicles))
        self.assertEqual('logfile', log_vehicles[0][0])
        self.assertEqual('vehicle_name', log_vehicles[0][1])

    def test_get_db_scenario_info(self) -> None:
        """
        Test the get_db_scenario_info query.
        """
        scenario_info_tags = list(get_db_scenario_info(self.db_file_name))
        self.assertEqual(2, len(scenario_info_tags))
        self.assertEqual('first_tag', scenario_info_tags[0][0])
        self.assertEqual(2, scenario_info_tags[0][1])
        self.assertEqual('second_tag', scenario_info_tags[1][0])
        self.assertEqual(1, scenario_info_tags[1][1])

class TestNuPlanScenarioQueries(unittest.TestCase):
    """
    Test suite for the NuPlan scenario queries.
    """
    generation_parameters: DBGenerationParameters

    @staticmethod
    def getDBFilePath() -> Path:
        """
        Get the location for the temporary SQLite file used for the test DB.
        :return: The filepath for the test data.
        """
        return Path('/tmp/test_nuplan_scenario_queries.sqlite3')

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create the mock DB data.
        """
        db_file_path = TestNuPlanScenarioQueries.getDBFilePath()
        if db_file_path.exists():
            db_file_path.unlink()
        cls.generation_parameters = DBGenerationParameters(num_lidars=1, num_cameras=2, num_sensor_data_per_sensor=50, num_lidarpc_per_image_ratio=2, num_scenes=10, num_traffic_lights_per_lidar_pc=5, num_agents_per_lidar_pc=3, num_static_objects_per_lidar_pc=2, scene_scenario_tag_mapping={5: ['first_tag'], 6: ['first_tag', 'second_tag'], 7: ['second_tag']}, file_path=str(db_file_path))
        generate_minimal_nuplan_db(cls.generation_parameters)

    def setUp(self) -> None:
        """
        The method to run before each test.
        """
        self.db_file_name = str(TestNuPlanScenarioQueries.getDBFilePath())
        self.sensor_source = SensorDataSource('lidar_pc', 'lidar', 'lidar_token', 'channel')

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Destroy the mock DB data.
        """
        db_file_path = TestNuPlanScenarioQueries.getDBFilePath()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)

    def test_get_sensor_token_from_index(self) -> None:
        """
        Test the get_sensor_token_from_index query.
        """
        for sample_index in [0, 12, 24]:
            retrieved_token = get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, sample_index)
            self.assertEqual(sample_index / self.generation_parameters.num_lidars, str_token_to_int(retrieved_token))
        self.assertIsNone(get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, 100000))
        with self.assertRaises(ValueError):
            get_sensor_token_by_index_from_db(self.db_file_name, self.sensor_source, -2)

    def test_get_end_sensor_time_from_db(self) -> None:
        """
        Test the get_end_sensor_time_from_db query.
        """
        log_end_time = get_end_sensor_time_from_db(self.db_file_name, sensor_source=self.sensor_source)
        self.assertEqual(49 * 1000000.0, log_end_time)

    def test_get_sensor_token_timestamp_from_db(self) -> None:
        """
        Test the get_sensor_data_token_timestamp_from_db query.
        """
        for token in [0, 3, 7]:
            expected_timestamp = token * 1000000.0
            actual_timestamp = get_sensor_data_token_timestamp_from_db(self.db_file_name, self.sensor_source, int_to_str_token(token))
            self.assertEqual(expected_timestamp, actual_timestamp)
        self.assertIsNone(get_sensor_data_token_timestamp_from_db(self.db_file_name, self.sensor_source, int_to_str_token(1000)))

    def test_get_sensor_token_map_name_from_db(self) -> None:
        """
        Test the get_sensor_token_map_name_from_db query.
        """
        for token in [0, 2, 6]:
            expected_map_name = 'map_version'
            actual_map_name = get_sensor_token_map_name_from_db(self.db_file_name, self.sensor_source, int_to_str_token(token))
            self.assertEqual(expected_map_name, actual_map_name)
        self.assertIsNone(get_sensor_token_map_name_from_db(self.db_file_name, self.sensor_source, int_to_str_token(1000)))

    def test_get_sampled_sensor_tokens_in_time_window_from_db(self) -> None:
        """
        Test the get_sampled_lidarpc_tokens_in_time_window_from_db query.
        """
        expected_tokens = [10, 13, 16, 19]
        actual_tokens = list((str_token_to_int(v) for v in get_sampled_sensor_tokens_in_time_window_from_db(log_file=self.db_file_name, sensor_source=self.sensor_source, start_timestamp=int(10 * 1000000.0), end_timestamp=int(20 * 1000000.0), subsample_interval=3)))
        self.assertEqual(expected_tokens, actual_tokens)

    def test_get_sensor_data_from_sensor_data_tokens_from_db(self) -> None:
        """
        Test the get_sensor_data_from_sensor_data_tokens_from_db query.
        """
        lidar_pc_tokens = [int_to_str_token(v) for v in [10, 13, 21]]
        image_tokens = [int_to_str_token(v) for v in [1100000]]
        lidar_pcs = [cast(LidarPc, sensor_data) for sensor_data in get_sensor_data_from_sensor_data_tokens_from_db(self.db_file_name, self.sensor_source, LidarPc, lidar_pc_tokens)]
        images = [cast(Image, sensor_data) for sensor_data in get_sensor_data_from_sensor_data_tokens_from_db(self.db_file_name, SensorDataSource('image', 'camera', 'camera_token', 'camera_0'), Image, image_tokens)]
        self.assertEqual(len(lidar_pc_tokens), len(lidar_pcs))
        self.assertEqual(len(image_tokens), len(images))
        lidar_pcs.sort(key=lambda x: int(x.timestamp))
        self.assertEqual(10, str_token_to_int(lidar_pcs[0].token))
        self.assertEqual(13, str_token_to_int(lidar_pcs[1].token))
        self.assertEqual(21, str_token_to_int(lidar_pcs[2].token))
        self.assertEqual(1100000, str_token_to_int(images[0].token))

    def test_get_lidar_transform_matrix_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_sensor_transform_matrix_for_sensor_data_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            xform_mat = get_sensor_transform_matrix_for_sensor_data_token_from_db(self.db_file_name, self.sensor_source, int_to_str_token(sample_token))
            self.assertIsNotNone(xform_mat)
            self.assertEqual(xform_mat[0, 3], 0)

    def test_get_mission_goal_for_sensor_data_token_from_db(self) -> None:
        """
        Test the get_mission_goal_for_sensor_data_token_from_db query.
        """
        query_lidarpc_token = int_to_str_token(12)
        expected_ego_pose_x = 14
        expected_ego_pose_y = 15
        result = get_mission_goal_for_sensor_data_token_from_db(self.db_file_name, self.sensor_source, query_lidarpc_token)
        self.assertIsNotNone(result)
        self.assertEqual(expected_ego_pose_x, result.x)
        self.assertEqual(expected_ego_pose_y, result.y)

    def test_get_roadblock_ids_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_roadblock_ids_for_lidarpc_token_from_db query.
        """
        result = get_roadblock_ids_for_lidarpc_token_from_db(self.db_file_name, int_to_str_token(0))
        self.assertEqual(result, ['0', '1', '2'])

    def test_get_statese2_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_statese2_for_lidarpc_token_from_db query.
        """
        query_lidarpc_token = int_to_str_token(13)
        expected_ego_pose_x = 13
        expected_ego_pose_y = 14
        result = get_statese2_for_lidarpc_token_from_db(self.db_file_name, query_lidarpc_token)
        self.assertIsNotNone(result)
        self.assertEqual(expected_ego_pose_x, result.x)
        self.assertEqual(expected_ego_pose_y, result.y)

    def test_get_sampled_lidarpcs_from_db(self) -> None:
        """
        Test the get_sampled_lidarpcs_from_db query.
        """
        test_cases = [{'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': True, 'expected_return_tokens': [5, 6, 7]}, {'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': False, 'expected_return_tokens': [3, 4, 5]}, {'initial_token': 7, 'sample_indexes': [0, 3, 12], 'future': False, 'expected_return_tokens': [4, 7]}, {'initial_token': 0, 'sample_indexes': [1000], 'future': True, 'expected_return_tokens': []}]
        for test_case in test_cases:
            initial_token = int_to_str_token(test_case['initial_token'])
            expected_return_tokens = [int_to_str_token(v) for v in test_case['expected_return_tokens']]
            actual_returned_lidarpcs = list(get_sampled_lidarpcs_from_db(self.db_file_name, initial_token, self.sensor_source, test_case['sample_indexes'], test_case['future']))
            self.assertEqual(len(expected_return_tokens), len(actual_returned_lidarpcs))
            for i in range(len(expected_return_tokens)):
                self.assertEqual(expected_return_tokens[i], actual_returned_lidarpcs[i].token)

    def test_get_sampled_ego_states_from_db(self) -> None:
        """
        Test the get_sampled_ego_states_from_db query.
        """
        test_cases = [{'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': True, 'expected_row_indexes': [5, 6, 7]}, {'initial_token': 5, 'sample_indexes': [0, 1, 2], 'future': False, 'expected_row_indexes': [3, 4, 5]}, {'initial_token': 7, 'sample_indexes': [0, 3, 12], 'future': False, 'expected_row_indexes': [4, 7]}, {'initial_token': 0, 'sample_indexes': [1000], 'future': True, 'expected_row_indexes': []}]
        for test_case in test_cases:
            initial_token = int_to_str_token(test_case['initial_token'])
            expected_row_indexes = test_case['expected_row_indexes']
            actual_returned_ego_states = list(get_sampled_ego_states_from_db(self.db_file_name, initial_token, self.sensor_source, test_case['sample_indexes'], test_case['future']))
            self.assertEqual(len(expected_row_indexes), len(actual_returned_ego_states))
            for i in range(len(expected_row_indexes)):
                self.assertEqual(expected_row_indexes[i] * 1000000.0, actual_returned_ego_states[i].time_point.time_us)

    def test_get_ego_state_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_ego_state_for_lidarpc_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            returned_pose = get_ego_state_for_lidarpc_token_from_db(self.db_file_name, query_token)
            self.assertEqual(sample_token * 1000000.0, returned_pose.time_point.time_us)

    def test_get_traffic_light_status_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_traffic_light_status_for_lidarpc_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            traffic_light_statuses = list(get_traffic_light_status_for_lidarpc_token_from_db(self.db_file_name, query_token))
            self.assertEqual(5, len(traffic_light_statuses))
            for tl_status in traffic_light_statuses:
                self.assertEqual(sample_token * 1000000.0, tl_status.timestamp)

    def test_get_tracked_objects_for_lidarpc_token_from_db(self) -> None:
        """
        Test the get_tracked_objects_for_token_from_db query.
        """
        for sample_token in [0, 30, 49]:
            query_token = int_to_str_token(sample_token)
            tracked_objects = list(get_tracked_objects_for_lidarpc_token_from_db(self.db_file_name, query_token))
            self.assertEqual(5, len(tracked_objects))
            agent_count = 0
            static_object_count = 0
            track_token_base_id = 600000
            token_base_id = 500000
            token_sample_step = 10000
            for idx, tracked_object in enumerate(tracked_objects):
                expected_track_token = track_token_base_id + idx
                expected_token = token_base_id + token_sample_step * sample_token + idx
                self.assertEqual(int_to_str_token(expected_track_token), tracked_object.track_token)
                self.assertEqual(int_to_str_token(expected_token), tracked_object.token)
                if isinstance(tracked_object, Agent):
                    agent_count += 1
                    self.assertEqual(TrackedObjectType.VEHICLE, tracked_object.tracked_object_type)
                    self.assertEqual(0, len(tracked_object.predictions))
                elif isinstance(tracked_object, StaticObject):
                    static_object_count += 1
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, tracked_object.tracked_object_type)
                else:
                    raise ValueError(f'Unexpected type: {type(tracked_object)}')
            self.assertEqual(3, agent_count)
            self.assertEqual(2, static_object_count)

    def test_get_tracked_objects_within_time_interval_from_db(self) -> None:
        """
        Test the get_tracked_objects_within_time_interval_from_db query.
        """
        expected_num_windows = {0: 3, 30: 5, 48: 4}
        expected_backward_offset = {0: 0, 30: -2, 48: -2}
        for sample_token in expected_num_windows.keys():
            start_timestamp = int(1000000.0 * (sample_token - 2))
            end_timestamp = int(1000000.0 * (sample_token + 2))
            tracked_objects = list(get_tracked_objects_within_time_interval_from_db(self.db_file_name, start_timestamp, end_timestamp, filter_track_tokens=None))
            expected_num_tokens = expected_num_windows[sample_token] * 5
            self.assertEqual(expected_num_tokens, len(tracked_objects))
            agent_count = 0
            static_object_count = 0
            track_token_base_id = 600000
            token_base_id = 500000
            token_sample_step = 10000
            for idx, tracked_object in enumerate(tracked_objects):
                expected_track_token = track_token_base_id + idx % 5
                expected_token = token_base_id + token_sample_step * (sample_token + expected_backward_offset[sample_token] + math.floor(idx / 5)) + idx % 5
                self.assertEqual(int_to_str_token(expected_track_token), tracked_object.track_token)
                self.assertEqual(int_to_str_token(expected_token), tracked_object.token)
                if isinstance(tracked_object, Agent):
                    agent_count += 1
                    self.assertEqual(TrackedObjectType.VEHICLE, tracked_object.tracked_object_type)
                    self.assertEqual(0, len(tracked_object.predictions))
                elif isinstance(tracked_object, StaticObject):
                    static_object_count += 1
                    self.assertEqual(TrackedObjectType.CZONE_SIGN, tracked_object.tracked_object_type)
                else:
                    raise ValueError(f'Unexpected type: {type(tracked_object)}')
            self.assertEqual(3 * expected_num_windows[sample_token], agent_count)
            self.assertEqual(2 * expected_num_windows[sample_token], static_object_count)

    def test_get_future_waypoints_for_agents_from_db(self) -> None:
        """
        Test the get_future_waypoints_for_agents_from_db query.
        """
        track_tokens = [600000, 600001, 600002]
        start_timestamp = 0
        end_timestamp = int(20 * 1000000.0 - 1)
        query_output: Dict[str, List[Waypoint]] = {}
        for token, waypoint in get_future_waypoints_for_agents_from_db(self.db_file_name, (int_to_str_token(t) for t in track_tokens), start_timestamp, end_timestamp):
            if token not in query_output:
                query_output[token] = []
            query_output[token].append(waypoint)
        expected_keys = ['{:08d}'.format(t) for t in track_tokens]
        self.assertEqual(len(expected_keys), len(query_output))
        for expected_key in expected_keys:
            self.assertTrue(expected_key in query_output)
            collected_waypoints = query_output[expected_key]
            self.assertEqual(20, len(collected_waypoints))
            for i in range(0, len(collected_waypoints), 1):
                self.assertEqual(i * 1000000.0, collected_waypoints[i].time_point.time_us)

    def test_get_scenarios_from_db(self) -> None:
        """
        Test the get_scenarios_from_db_query.
        """
        no_filter_output: List[int] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            no_filter_output.append(str_token_to_int(row['token'].hex()))
        self.assertEqual(list(range(10, 40, 1)), no_filter_output)
        filter_tokens = [int_to_str_token(v) for v in [15, 30]]
        tokens_filter_output: List[int] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=filter_tokens, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            tokens_filter_output.append(row['token'].hex())
        self.assertEqual(filter_tokens, tokens_filter_output)
        filter_scenarios = ['first_tag']
        extracted_rows: List[Tuple[int, str]] = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=filter_scenarios, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            extracted_rows.append((str_token_to_int(row['token'].hex()), row['scenario_type']))
        self.assertEqual(2, len(extracted_rows))
        self.assertEqual(25, extracted_rows[0][0])
        self.assertEqual('first_tag', extracted_rows[0][1])
        self.assertEqual(30, extracted_rows[1][0])
        self.assertEqual('first_tag', extracted_rows[1][1])
        filter_scenarios = ['second_tag']
        extracted_rows = []
        for row in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=filter_scenarios, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=False):
            extracted_rows.append((str_token_to_int(row['token'].hex()), row['scenario_type']))
        self.assertEqual(2, len(extracted_rows))
        self.assertEqual(30, extracted_rows[0][0])
        self.assertEqual('second_tag', extracted_rows[0][1])
        self.assertEqual(35, extracted_rows[1][0])
        self.assertEqual('second_tag', extracted_rows[1][1])
        filter_maps = ['map_version']
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=filter_maps, include_invalid_mission_goals=False, include_cameras=False)))
        self.assertLess(0, row_cnt)
        filter_maps = ['map_that_does_not_exist']
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=filter_maps, include_invalid_mission_goals=False, include_cameras=False)))
        self.assertEqual(0, row_cnt)
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=None, filter_types=None, filter_map_names=None, include_invalid_mission_goals=False, include_cameras=True)))
        self.assertEqual(15, row_cnt)
        row_cnt = sum((1 for _ in get_scenarios_from_db(self.db_file_name, filter_tokens=[int_to_str_token(25)], filter_types=['first_tag'], filter_map_names=['map_version'], include_invalid_mission_goals=False, include_cameras=False)))
        self.assertEqual(1, row_cnt)

    def test_get_lidarpc_tokens_with_scenario_tag_from_db(self) -> None:
        """
        Test the get_lidarpc_tokens_with_scenario_tag_from_db query.
        """
        tuples = list(get_lidarpc_tokens_with_scenario_tag_from_db(self.db_file_name))
        self.assertEqual(4, len(tuples))
        expected_tuples = [('first_tag', int_to_str_token(25)), ('first_tag', int_to_str_token(30)), ('second_tag', int_to_str_token(30)), ('second_tag', int_to_str_token(35))]
        for tup in tuples:
            self.assertTrue(tup in expected_tuples)

    def test_get_sensor_token(self) -> None:
        """Test the get_lidarpc_token_from_index query."""
        retrieved_token = get_sensor_token(self.db_file_name, 'lidar', 'channel')
        self.assertEqual(700000, str_token_to_int(retrieved_token))
        with self.assertRaisesRegex(RuntimeError, 'Channel missing_channel not found in table lidar!'):
            self.assertIsNone(get_sensor_token(self.db_file_name, 'lidar', 'missing_channel'))

    def test_get_images_from_lidar_tokens(self) -> None:
        """Test the get_images_from_lidar_tokens query."""
        token = int_to_str_token(20)
        retrieved_images = list(get_images_from_lidar_tokens(self.db_file_name, [token], ['camera_0', 'camera_1'], 50000, 50000))
        self.assertEqual(2, len(retrieved_images))
        self.assertEqual(1100020, str_token_to_int(retrieved_images[0].token))
        self.assertEqual(1100070, str_token_to_int(retrieved_images[1].token))
        self.assertEqual('camera_0', retrieved_images[0].channel)
        self.assertEqual('camera_1', retrieved_images[1].channel)

    def test_get_cameras(self) -> None:
        """Test the get_cameras query."""
        retrieved_cameras = list(get_cameras(self.db_file_name, ['camera_0', 'camera_1']))
        self.assertEqual(2, len(retrieved_cameras))
        self.assertEqual(1000000, str_token_to_int(retrieved_cameras[0].token))
        self.assertEqual(1000001, str_token_to_int(retrieved_cameras[1].token))
        self.assertEqual('camera_0', retrieved_cameras[0].channel)
        self.assertEqual('camera_1', retrieved_cameras[1].channel)
        retrieved_cameras = list(get_cameras(self.db_file_name, ['camera_1']))
        self.assertEqual(1, len(retrieved_cameras))
        self.assertEqual(1000001, str_token_to_int(retrieved_cameras[0].token))
        self.assertEqual('camera_1', retrieved_cameras[0].channel)


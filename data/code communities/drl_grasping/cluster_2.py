# Cluster 2

def generate_launch_description() -> LaunchDescription:
    declared_arguments = generate_declared_arguments()
    robot_model = LaunchConfiguration('robot_model')
    robot_name = LaunchConfiguration('robot_name')
    prefix = LaunchConfiguration('prefix')
    env = LaunchConfiguration('env')
    env_kwargs = LaunchConfiguration('env_kwargs')
    algo = LaunchConfiguration('algo')
    hyperparams = LaunchConfiguration('hyperparams')
    n_timesteps = LaunchConfiguration('n_timesteps')
    num_threads = LaunchConfiguration('num_threads')
    seed = LaunchConfiguration('seed')
    trained_agent = LaunchConfiguration('trained_agent')
    save_freq = LaunchConfiguration('save_freq')
    save_replay_buffer = LaunchConfiguration('save_replay_buffer')
    preload_replay_buffer = LaunchConfiguration('preload_replay_buffer')
    log_folder = LaunchConfiguration('log_folder')
    tensorboard_log = LaunchConfiguration('tensorboard_log')
    log_interval = LaunchConfiguration('log_interval')
    uuid = LaunchConfiguration('uuid')
    eval_freq = LaunchConfiguration('eval_freq')
    eval_episodes = LaunchConfiguration('eval_episodes')
    verbose = LaunchConfiguration('verbose')
    truncate_last_trajectory = LaunchConfiguration('truncate_last_trajectory')
    enable_rviz = LaunchConfiguration('enable_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')
    launch_descriptions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('drl_grasping'), 'launch', 'sim', 'sim.launch.py'])), launch_arguments=[('robot_model', robot_model), ('robot_name', robot_name), ('prefix', prefix), ('enable_rviz', enable_rviz), ('rviz_config', rviz_config), ('use_sim_time', use_sim_time), ('log_level', log_level)])]
    nodes = [Node(package='drl_grasping', executable='train.py', output='log', arguments=['--env', env, '--env-kwargs', env_kwargs, '--env-kwargs', ['robot_model:"', robot_model, '"'], '--algo', algo, '--hyperparams', hyperparams, '--n-timesteps', n_timesteps, '--num-threads', num_threads, '--seed', seed, '--trained-agent', trained_agent, '--save-freq', save_freq, '--save-replay-buffer', save_replay_buffer, '--preload-replay-buffer', preload_replay_buffer, '--log-folder', log_folder, '--tensorboard-log', tensorboard_log, '--log-interval', log_interval, '--uuid', uuid, '--eval-freq', eval_freq, '--eval-episodes', eval_episodes, '--verbose', verbose, '--truncate-last-trajectory', truncate_last_trajectory, '--ros-args', '--log-level', log_level], parameters=[{'use_sim_time': use_sim_time}])]
    environment_variables = [SetEnvironmentVariable(name='OMP_DYNAMIC', value='TRUE'), SetEnvironmentVariable(name='OMP_NUM_THREADS', value=str(cpu_count() // 2))]
    return LaunchDescription(declared_arguments + launch_descriptions + nodes + environment_variables)

def generate_declared_arguments() -> List[DeclareLaunchArgument]:
    """
    Generate list of all launch arguments that are declared for this launch script.
    """
    return [DeclareLaunchArgument('robot_model', default_value='lunalab_summit_xl_gen', description="Name of the robot to use. Supported options are: 'panda' and 'lunalab_summit_xl_gen'."), DeclareLaunchArgument('robot_name', default_value=LaunchConfiguration('robot_model'), description='Name of the robot.'), DeclareLaunchArgument('prefix', default_value='robot_', description='Prefix for all robot entities. If modified, then joint names in the configuration of controllers must also be updated.'), DeclareLaunchArgument('env', default_value='GraspPlanetary-OctreeWithColor-Gazebo-v0', description='Environment ID'), DeclareLaunchArgument('env_kwargs', default_value=['robot_model:"', LaunchConfiguration('robot_model'), '"'], description='Optional keyword argument to pass to the env constructor.'), DeclareLaunchArgument('n_episodes', default_value='1000', description='Overwrite the number of episodes.'), DeclareLaunchArgument('seed', default_value='69', description='Random generator seed.'), DeclareLaunchArgument('check_env', default_value='True', description='Flag to check the environment before running the random agent.'), DeclareLaunchArgument('render', default_value='True', description='Flag to enable rendering.'), DeclareLaunchArgument('enable_rviz', default_value='true', description='Flag to enable RViz2.'), DeclareLaunchArgument('rviz_config', default_value=path.join(get_package_share_directory('drl_grasping'), 'rviz', 'drl_grasping.rviz'), description='Path to configuration for RViz2.'), DeclareLaunchArgument('use_sim_time', default_value='true', description='If true, use simulated clock.'), DeclareLaunchArgument('log_level', default_value='error', description='The level of logging that is applied to all ROS 2 nodes launched by this script.')]

def generate_launch_description() -> LaunchDescription:
    declared_arguments = generate_declared_arguments()
    robot_model = LaunchConfiguration('robot_model')
    robot_name = LaunchConfiguration('robot_name')
    prefix = LaunchConfiguration('prefix')
    env = LaunchConfiguration('env')
    env_kwargs = LaunchConfiguration('env_kwargs')
    algo = LaunchConfiguration('algo')
    n_timesteps = LaunchConfiguration('n_timesteps')
    num_threads = LaunchConfiguration('num_threads')
    seed = LaunchConfiguration('seed')
    preload_replay_buffer = LaunchConfiguration('preload_replay_buffer')
    log_folder = LaunchConfiguration('log_folder')
    tensorboard_log = LaunchConfiguration('tensorboard_log')
    log_interval = LaunchConfiguration('log_interval')
    uuid = LaunchConfiguration('uuid')
    sampler = LaunchConfiguration('sampler')
    pruner = LaunchConfiguration('pruner')
    n_trials = LaunchConfiguration('n_trials')
    n_startup_trials = LaunchConfiguration('n_startup_trials')
    n_evaluations = LaunchConfiguration('n_evaluations')
    n_jobs = LaunchConfiguration('n_jobs')
    storage = LaunchConfiguration('storage')
    study_name = LaunchConfiguration('study_name')
    eval_episodes = LaunchConfiguration('eval_episodes')
    verbose = LaunchConfiguration('verbose')
    truncate_last_trajectory = LaunchConfiguration('truncate_last_trajectory')
    enable_rviz = LaunchConfiguration('enable_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')
    launch_descriptions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('drl_grasping'), 'launch', 'sim', 'sim.launch.py'])), launch_arguments=[('robot_model', robot_model), ('robot_name', robot_name), ('prefix', prefix), ('enable_rviz', enable_rviz), ('rviz_config', rviz_config), ('use_sim_time', use_sim_time), ('log_level', log_level)])]
    nodes = [Node(package='drl_grasping', executable='train.py', output='log', arguments=['--env', env, '--env-kwargs', env_kwargs, '--env-kwargs', ['robot_model:"', robot_model, '"'], '--algo', algo, '--seed', seed, '--num-threads', num_threads, '--n-timesteps', n_timesteps, '--preload-replay-buffer', preload_replay_buffer, '--log-folder', log_folder, '--tensorboard-log', tensorboard_log, '--log-interval', log_interval, '--uuid', uuid, '--optimize-hyperparameters', 'True', '--sampler', sampler, '--pruner', pruner, '--n-trials', n_trials, '--n-startup-trials', n_startup_trials, '--n-evaluations', n_evaluations, '--n-jobs', n_jobs, '--storage', storage, '--study-name', study_name, '--eval-episodes', eval_episodes, '--verbose', verbose, '--truncate-last-trajectory', truncate_last_trajectory, '--ros-args', '--log-level', log_level], parameters=[{'use_sim_time': use_sim_time}])]
    environment_variables = [SetEnvironmentVariable(name='OMP_DYNAMIC', value='TRUE'), SetEnvironmentVariable(name='OMP_NUM_THREADS', value=str(cpu_count() // 2))]
    return LaunchDescription(declared_arguments + launch_descriptions + nodes + environment_variables)

def generate_launch_description() -> LaunchDescription:
    declared_arguments = generate_declared_arguments()
    robot_model = LaunchConfiguration('robot_model')
    robot_name = LaunchConfiguration('robot_name')
    prefix = LaunchConfiguration('prefix')
    env = LaunchConfiguration('env')
    env_kwargs = LaunchConfiguration('env_kwargs')
    algo = LaunchConfiguration('algo')
    num_threads = LaunchConfiguration('num_threads')
    n_episodes = LaunchConfiguration('n_episodes')
    seed = LaunchConfiguration('seed')
    log_folder = LaunchConfiguration('log_folder')
    exp_id = LaunchConfiguration('exp_id')
    load_best = LaunchConfiguration('load_best')
    load_checkpoint = LaunchConfiguration('load_checkpoint')
    stochastic = LaunchConfiguration('stochastic')
    reward_log = LaunchConfiguration('reward_log')
    norm_reward = LaunchConfiguration('norm_reward')
    no_render = LaunchConfiguration('no_render')
    verbose = LaunchConfiguration('verbose')
    enable_rviz = LaunchConfiguration('enable_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')
    launch_descriptions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('drl_grasping'), 'launch', 'sim', 'sim.launch.py'])), launch_arguments=[('robot_model', robot_model), ('robot_name', robot_name), ('prefix', prefix), ('enable_rviz', enable_rviz), ('rviz_config', rviz_config), ('use_sim_time', use_sim_time), ('log_level', log_level)])]
    nodes = [Node(package='drl_grasping', executable='evaluate.py', output='log', arguments=['--env', env, '--env-kwargs', env_kwargs, '--env-kwargs', ['robot_model:"', robot_model, '"'], '--algo', algo, '--seed', seed, '--num-threads', num_threads, '--n-episodes', n_episodes, '--log-folder', log_folder, '--exp-id', exp_id, '--load-best', load_best, '--load-checkpoint', load_checkpoint, '--stochastic', stochastic, '--reward-log', reward_log, '--norm-reward', norm_reward, '--no-render', no_render, '--verbose', verbose, '--ros-args', '--log-level', log_level], parameters=[{'use_sim_time': use_sim_time}])]
    return LaunchDescription(declared_arguments + launch_descriptions + nodes)

def generate_launch_description() -> LaunchDescription:
    declared_arguments = generate_declared_arguments()
    robot_model = LaunchConfiguration('robot_model')
    robot_name = LaunchConfiguration('robot_name')
    prefix = LaunchConfiguration('prefix')
    env = LaunchConfiguration('env')
    env_kwargs = LaunchConfiguration('env_kwargs')
    seed = LaunchConfiguration('seed')
    log_folder = LaunchConfiguration('log_folder')
    eval_freq = LaunchConfiguration('eval_freq')
    verbose = LaunchConfiguration('verbose')
    enable_rviz = LaunchConfiguration('enable_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')
    launch_descriptions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('drl_grasping'), 'launch', 'sim', 'sim.launch.py'])), launch_arguments=[('robot_model', robot_model), ('robot_name', robot_name), ('prefix', prefix), ('enable_rviz', enable_rviz), ('rviz_config', rviz_config), ('use_sim_time', use_sim_time), ('log_level', log_level)])]
    nodes = [Node(package='drl_grasping', executable='train_dreamerv2.py', output='log', arguments=['--env', env, '--env-kwargs', env_kwargs, '--env-kwargs', ['robot_model:"', robot_model, '"'], '--seed', seed, '--log-folder', log_folder, '--eval-freq', eval_freq, '--verbose', verbose, '--ros-args', '--log-level', log_level], parameters=[{'use_sim_time': use_sim_time}])]
    return LaunchDescription(declared_arguments + launch_descriptions + nodes)

def generate_launch_description() -> LaunchDescription:
    declared_arguments = generate_declared_arguments()
    world_name = LaunchConfiguration('world_name')
    robot_model = LaunchConfiguration('robot_model')
    robot_name = LaunchConfiguration('robot_name')
    enable_rviz = LaunchConfiguration('enable_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')
    declared_arguments.append(DeclareLaunchArgument('__prefix', default_value='panda_', description='Robot-specific prefix for panda.', condition=LaunchConfigurationEquals('robot_model', 'panda')))
    declared_arguments.append(DeclareLaunchArgument('__prefix', default_value=LaunchConfiguration('prefix'), description='Robot-specific prefix for all other robots.', condition=LaunchConfigurationNotEquals('robot_model', 'panda')))
    prefix = LaunchConfiguration('__prefix')
    launch_descriptions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare([robot_model, '_moveit_config']), 'launch', 'move_group.launch.py'])), launch_arguments=[('name', robot_name), ('prefix', prefix), ('enable_rviz', enable_rviz), ('rviz_config', rviz_config), ('use_sim_time', use_sim_time), ('log_level', log_level)]), IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('lunalab_summit_xl_gen_ign'), 'launch', 'bridge.launch.py'])), launch_arguments=[('world_name', world_name), ('robot_name', robot_name), ('prefix', prefix), ('use_sim_time', use_sim_time), ('log_level', log_level)], condition=LaunchConfigurationEquals('robot_model', 'lunalab_summit_xl_gen'))]
    nodes = [Node(package='ros_ign_bridge', executable='parameter_bridge', output='log', arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock', '--ros-args', '--log-level', log_level], parameters=[{'use_sim_time': use_sim_time}], condition=LaunchConfigurationNotEquals('robot_model', 'lunalab_summit_xl_gen')), Node(package='tf2_ros', executable='static_transform_publisher', output='log', arguments=['0', '0', '0', '0', '0', '0', world_name, [prefix, 'link0'], '--ros-args', '--log-level', log_level], parameters=[{'use_sim_time': use_sim_time}], condition=LaunchConfigurationEquals('robot_model', 'panda'))]
    logs = [LogInfo(msg=['Configuring drl_grasping for Ignition Gazebo world ', world_name, '\n\tRobot model: ', robot_name, '\n\tPrefix: ', prefix])]
    return LaunchDescription(declared_arguments + launch_descriptions + nodes + logs)

def generate_launch_description() -> LaunchDescription:
    declared_arguments = generate_declared_arguments()
    robot_name = LaunchConfiguration('robot_name')
    prefix = LaunchConfiguration('prefix')
    enable_rviz = LaunchConfiguration('enable_rviz')
    rviz_config = LaunchConfiguration('rviz_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level')
    launch_descriptions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare(['lunalab_summit_xl_gen_moveit_config']), 'launch', 'move_group_ros1_controllers.launch.py'])), launch_arguments=[('name', robot_name), ('prefix', prefix), ('enable_rviz', enable_rviz), ('rviz_config', rviz_config), ('use_sim_time', use_sim_time), ('log_level', log_level)])]
    nodes = [Node(package='tf2_ros', executable='static_transform_publisher', output='log', arguments=['0', '0', '0', '0', '0', '0', 'drl_grasping_world', [prefix, 'summit_xl_base_footprint'], '--ros-args', '--log-level', log_level], parameters=[{'use_sim_time': use_sim_time}]), Node(package='tf2_ros', executable='static_transform_publisher', output='log', arguments=['0.222518', '0.0152645', '0.0862207', '-0.00926349', '0.730585', '0.0803686', [prefix, 'j2s7s300_link_base'], 'rs_d435', '--ros-args', '--log-level', log_level], parameters=[{'use_sim_time': use_sim_time}])]
    logs = [LogInfo(msg=['Configuring drl_grasping for real Summit XL-GEN (LunaLab variant)'])]
    return LaunchDescription(declared_arguments + launch_descriptions + nodes + logs)


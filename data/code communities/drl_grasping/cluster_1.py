# Cluster 1

def main(args: Dict):
    registered_envs = set(gym.envs.registry.env_specs.keys())
    if args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(args.env, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f'{args.env} not found in gym registry, you maybe meant {closest_match}?')
    if args.seed < 0:
        args.seed = np.random.randint(2 ** 32 - 1, dtype=np.int64).item()
    set_random_seed(args.seed)
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f'Setting torch.num_threads to {args.num_threads}')
        th.set_num_threads(args.num_threads)
    if args.trained_agent != '':
        assert args.trained_agent.endswith('.zip') and os.path.isfile(args.trained_agent), 'The trained_agent must be a valid path to a .zip file'
    uuid_str = f'_{uuid.uuid4()}' if args.uuid else ''
    env_kwargs = args.env_kwargs
    env_kwargs.update({'preload_replay_buffer': True})
    print('=' * 10, args.env, '=' * 10)
    print(f'Seed: {args.seed}')
    exp_manager = ExperimentManager(args, args.algo, args.env, args.log_folder, args.tensorboard_log, args.n_timesteps, args.eval_freq, args.eval_episodes, args.save_freq, args.hyperparams, args.env_kwargs, args.trained_agent, truncate_last_trajectory=args.truncate_last_trajectory, uuid_str=uuid_str, seed=args.seed, log_interval=args.log_interval, save_replay_buffer=args.save_replay_buffer, verbose=args.verbose, vec_env_type=args.vec_env)
    model = exp_manager.setup_experiment()
    exp_manager.collect_demonstration(model)

def main(args: Dict):
    registered_envs = set(gym.envs.registry.env_specs.keys())
    if args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(args.env, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f'{args.env} not found in gym registry, you maybe meant {closest_match}?')
    if args.seed < 0:
        args.seed = np.random.randint(2 ** 32 - 1, dtype=np.int64).item()
    set_random_seed(args.seed)
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f'Setting torch.num_threads to {args.num_threads}')
        th.set_num_threads(args.num_threads)
    if args.trained_agent != '':
        assert args.trained_agent.endswith('.zip') and os.path.isfile(args.trained_agent), 'The trained_agent must be a valid path to a .zip file'
    uuid_str = f'_{uuid.uuid4()}' if args.uuid else ''
    print('=' * 10, args.env, '=' * 10)
    print(f'Seed: {args.seed}')
    exp_manager = ExperimentManager(args, args.algo, args.env, args.log_folder, args.tensorboard_log, args.n_timesteps, args.eval_freq, args.eval_episodes, args.save_freq, args.hyperparams, args.env_kwargs, args.trained_agent, truncate_last_trajectory=args.truncate_last_trajectory, uuid_str=uuid_str, seed=args.seed, log_interval=args.log_interval, save_replay_buffer=args.save_replay_buffer, preload_replay_buffer=args.preload_replay_buffer, verbose=args.verbose, vec_env_type=args.vec_env)
    model = exp_manager.setup_experiment()
    exp_manager.learn(model)
    exp_manager.save_trained_model(model)


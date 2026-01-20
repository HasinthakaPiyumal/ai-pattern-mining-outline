# Cluster 0

def main(args: Dict):
    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(args.log_folder, args.algo), args.env)
        print(f'Loading latest experiment, id={args.exp_id}')
    if args.exp_id > 0:
        log_path = os.path.join(args.log_folder, args.algo, f'{args.env}_{args.exp_id}')
    else:
        log_path = os.path.join(args.log_folder, args.algo)
    assert os.path.isdir(log_path), f'The {log_path} folder was not found'
    found = False
    for ext in ['zip']:
        model_path = os.path.join(log_path, f'{args.env}.{ext}')
        found = os.path.isfile(model_path)
        if found:
            break
    if args.load_best:
        model_path = os.path.join(log_path, 'best_model.zip')
        found = os.path.isfile(model_path)
    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f'rl_model_{args.load_checkpoint}_steps.zip')
        found = os.path.isfile(model_path)
    if not found:
        raise ValueError(f'No model found for {args.algo} on {args.env}, path: {model_path}')
    off_policy_algos = ['qrdqn', 'dqn', 'ddpg', 'sac', 'her', 'td3', 'tqc']
    if args.algo in off_policy_algos:
        args.n_envs = 1
    set_random_seed(args.seed)
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f'Setting torch.num_threads to {args.num_threads}')
        th.set_num_threads(args.num_threads)
    stats_path = os.path.join(log_path, args.env)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)
    env_kwargs = {}
    args_path = os.path.join(log_path, args.env, 'args.yml')
    if os.path.isfile(args_path):
        with open(args_path, 'r') as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args['env_kwargs'] is not None:
                env_kwargs = loaded_args['env_kwargs']
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)
    log_dir = args.reward_log if args.reward_log != '' else None
    env = create_test_env(args.env, n_envs=args.n_envs, stats_path=stats_path, seed=args.seed, log_dir=log_dir, should_render=not args.no_render, hyperparams=hyperparams, env_kwargs=env_kwargs)
    kwargs = dict(seed=args.seed)
    if args.algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))
    model = ALGOS[args.algo].load(model_path, env=env, **kwargs)
    obs = env.reset()
    stochastic = args.stochastic
    deterministic = not stochastic
    print(f'Evaluating for {args.n_episodes} episodes with a', 'deterministic' if deterministic else 'stochastic', 'policy.')
    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths, success_episode_lengths = ([], [], [])
    ep_len = 0
    episode = 0
    successes = []
    while episode < args.n_episodes:
        action, state = model.predict(obs, state=state, deterministic=deterministic)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render('human')
        episode_reward += reward[0]
        ep_len += 1
        if done and args.verbose > 0:
            episode += 1
            print(f'--- Episode {episode}/{args.n_episodes}')
            print(f'Episode Reward: {episode_reward:.2f}')
            episode_rewards.append(episode_reward)
            print('Episode Length', ep_len)
            episode_lengths.append(ep_len)
            if infos[0].get('is_success') is not None:
                print('Success?:', infos[0].get('is_success', False))
                successes.append(infos[0].get('is_success', False))
                if infos[0].get('is_success'):
                    success_episode_lengths.append(ep_len)
                print(f'Current success rate: {100 * np.mean(successes):.2f}%')
            episode_reward = 0.0
            ep_len = 0
            state = None
    if args.verbose > 0 and len(successes) > 0:
        print(f'Success rate: {100 * np.mean(successes):.2f}%')
    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f'Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}')
    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f'Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}')
    if args.verbose > 0 and len(success_episode_lengths) > 0:
        print(f'Mean episode length of successful episodes: {np.mean(success_episode_lengths):.2f} +/- {np.std(success_episode_lengths):.2f}')
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in args.env and isinstance(env, VecEnv):
            while isinstance(env, VecEnvWrapper):
                env = env.venv
            if isinstance(env, DummyVecEnv):
                env.envs[0].env.close()
            else:
                env.close()
        else:
            env.close()


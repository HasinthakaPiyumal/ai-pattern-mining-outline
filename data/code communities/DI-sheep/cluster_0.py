# Cluster 0

def main(input_cfg, seed, max_env_step=int(1000000.0), max_train_iter=int(1000000.0)):
    cfg, create_cfg = input_cfg
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)
    collector_env = create_env_manager(cfg.env.manager, [lambda: sheep_env_fn(cfg.env.level) for _ in range(cfg.env.collector_env_num)])
    evaluator_env = create_env_manager(cfg.env.manager, [lambda: sheep_env_fn(cfg.env.level) for _ in range(cfg.env.evaluator_env_num)])
    collector_env.seed(cfg.seed, dynamic_seed=False)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    obs_space = collector_env._env_ref.observation_space
    model = SheepModel(obs_space['item_obs'].shape[1], obs_space['item_obs'].shape[0], 'TF', obs_space['bucket_obs'].shape[0], obs_space['global_obs'].shape[0])
    policy = PPOPolicy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval'])
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(cfg.policy.collect.collector, env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    learner.call_hook('before_run')
    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break
    learner.call_hook('after_run')


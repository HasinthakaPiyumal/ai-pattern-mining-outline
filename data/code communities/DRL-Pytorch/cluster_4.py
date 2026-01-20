# Cluster 4

def main():
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    render_mode = 'human' if opt.render else None
    eval_env = make_env_tianshou(opt.EnvName, noop_reset=opt.noop_reset, episode_life=False, clip_rewards=False, render_mode=render_mode)
    opt.action_dim = eval_env.action_space.n
    print('Algorithm:', opt.algo_name, '  Env:', opt.EnvName, '  Action_dim:', opt.action_dim, '  Seed:', opt.seed, '\n')
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = DeepQ_Agent(opt)
    if opt.Loadmodel:
        agent.load(opt.ExperimentName, opt.ModelIdex)
    if opt.render:
        while True:
            score = evaluate_policy(eval_env, agent, seed=opt.seed, turns=1)
            print(opt.ExperimentName, 'seed:', opt.seed, 'score:', score)
    else:
        if opt.write:
            from torch.utils.tensorboard import SummaryWriter
            timenow = str(datetime.now())[0:-7]
            timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2:]
            writepath = f'runs/{opt.ExperimentName}_S{opt.seed}' + timenow
            if os.path.exists(writepath):
                shutil.rmtree(writepath)
            writer = SummaryWriter(log_dir=writepath)
        buffer = ReplayBuffer_torch(device=opt.dvc, max_size=opt.buffersize)
        env = make_env_tianshou(opt.EnvName, noop_reset=opt.noop_reset)
        schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=opt.final_e, initial_p=opt.init_e)
        agent.exp_noise = opt.init_e
        seed = opt.seed
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=seed)
            seed += 1
            done = False
            while not done:
                a = agent.select_action(s, evaluate=False)
                s_next, r, dw, tr, info = env.step(a)
                buffer.add(s, a, r, s_next, dw)
                done = dw + tr
                s = s_next
                if buffer.size >= opt.random_steps:
                    agent.train(buffer)
                    'record & log'
                    if total_steps % opt.eval_interval == 0:
                        score = evaluate_policy(eval_env, agent, seed=seed + 1)
                        if opt.write:
                            writer.add_scalar('ep_r', score, global_step=total_steps)
                            writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                        print(f'{opt.ExperimentName}, Seed:{opt.seed}, Step:{int(total_steps / 1000)}k, Score:{score}')
                        agent.exp_noise = schedualer.value(total_steps)
                    total_steps += 1
                    'save model'
                    if total_steps % opt.save_interval == 0:
                        agent.save(opt.ExperimentName, int(total_steps / 1000))
    env.close()
    eval_env.close()

def make_env_tianshou(env_name, noop_reset=True, episode_life=True, clip_rewards=True, frame_stack=4, warp_frame=True, render_mode=None):
    """Configure environment for DeepMind-style Atari.
    Support both Gymnasium(s,r,term,trunc,info) and Gym(s,r,done,info)  API
    The observation is (4, 84, 84); torch.uint8; <class 'torch.Tensor'>
    # Here we do not normalize the observation to float in (0,1). Instead, we use uint8 to save memory.
    """
    assert 'NoFrameskip' in env_name
    env = gym.make(env_name, render_mode=render_mode)
    if noop_reset:
        env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if warp_frame:
        env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env


# Cluster 0

def main():
    EnvName = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'Humanoid-v4', 'HalfCheetah-v4', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4', 'BWv3', 'BWHv3']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{env._max_episode_steps}')
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed: {}'.format(opt.seed))
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2:]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = DDPG_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)
    if opt.render:
        while True:
            score = evaluate_policy(env, agent, turns=1)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False
            'Interact & trian'
            while not done:
                if total_steps < opt.random_steps:
                    a = env.action_space.sample()
                else:
                    a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a)
                done = dw or tr
                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1
                'train'
                if total_steps >= opt.random_steps:
                    agent.train()
                'record & log'
                if total_steps % opt.eval_interval == 0:
                    ep_r = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps / 1000)}k, Episode Reward:{ep_r}')
                'save model'
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps / 1000))
        env.close()
        eval_env.close()


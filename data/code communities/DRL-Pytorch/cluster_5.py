# Cluster 5

def main():
    EnvName = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'Humanoid-v4', 'HalfCheetah-v4', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4', 'BWv3', 'BWHv3']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    opt.max_e_steps = env._max_episode_steps
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')
    env_seed = opt.seed
    np.random.seed(opt.seed)
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
    agent = TD3_agent(**vars(opt))
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
                if total_steps < 10 * opt.max_e_steps:
                    a = env.action_space.sample()
                else:
                    a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a)
                r = Reward_adapter(r, opt.EnvIdex)
                done = dw or tr
                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1
                'train if its time'
                if total_steps >= 2 * opt.max_e_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        agent.train()
                'record & log'
                if total_steps % opt.eval_interval == 0:
                    agent.explore_noise *= opt.explore_noise_decay
                    ep_r = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps / 1000)}k, Episode Reward:{ep_r}')
                'save model'
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps / 1000))
        env.close()
        eval_env.close()

def Reward_adapter(r, EnvIdex):
    if EnvIdex == 0:
        r = (r + 8) / 8
    elif EnvIdex == 1:
        if r <= -100:
            r = -10
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100:
            r = -1
    return r

def evaluate_policy(env, agent, max_action, turns):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            a, logprob_a = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)
            s_next, r, dw, tr, info = env.step(act)
            done = dw or tr
            total_scores += r
            s = s_next
    return total_scores / turns

def Action_adapter(a, max_action):
    return 2 * (a - 0.5) * max_action

def main():
    EnvName = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'Humanoid-v4', 'HalfCheetah-v4', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4', 'BWv3', 'BWHv3']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    opt.max_steps = env._max_episode_steps
    print('Env:', EnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '  max_a:', opt.max_action, '  min_a:', env.action_space.low[0], 'max_steps', opt.max_steps)
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
    agent = PPO_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)
    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, opt.max_action, 1)
            print(f'Env:{EnvName[opt.EnvIdex]}, Episode Reward:{ep_r}')
    else:
        traj_lenth, total_steps = (0, 0)
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False
            'Interact & trian'
            while not done:
                'Interact with Env'
                a, logprob_a = agent.select_action(s, deterministic=False)
                act = Action_adapter(a, opt.max_action)
                s_next, r, dw, tr, info = env.step(act)
                r = Reward_adapter(r, opt.EnvIdex)
                done = dw or tr
                'Store the current transition'
                agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx=traj_lenth)
                s = s_next
                traj_lenth += 1
                total_steps += 1
                'Update if its time'
                if traj_lenth % opt.T_horizon == 0:
                    agent.train()
                    traj_lenth = 0
                'Record & log'
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, opt.max_action, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:', EnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', score)
                'Save model'
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps / 1000))
        env.close()
        eval_env.close()

def evaluate_policy(env, max_action, agent, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            a = agent.select_action(s, deterministic=True)
            act = Action_adapter(a, max_action)
            s_next, r, dw, tr, info = env.step(act)
            done = dw or tr
            total_scores += r
            s = s_next
    return int(total_scores / turns)

def main():
    EnvName = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'Humanoid-v4', 'HalfCheetah-v4', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4', 'BWv3', 'BWHv3']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    opt.max_e_steps = env._max_episode_steps
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')
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
    agent = SAC_countinuous(**vars(opt))
    if opt.Loadmodel:
        agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)
    if opt.render:
        while True:
            score = evaluate_policy(env, opt.max_action, agent, turns=1)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False
            'Interact & trian'
            while not done:
                if total_steps < 5 * opt.max_e_steps:
                    act = env.action_space.sample()
                    a = Action_adapter_reverse(act, opt.max_action)
                else:
                    a = agent.select_action(s, deterministic=False)
                    act = Action_adapter(a, opt.max_action)
                s_next, r, dw, tr, info = env.step(act)
                r = Reward_adapter(r, opt.EnvIdex)
                done = dw or tr
                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1
                "train if it's time"
                if total_steps >= 2 * opt.max_e_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        agent.train()
                'record & log'
                if total_steps % opt.eval_interval == 0:
                    ep_r = evaluate_policy(eval_env, opt.max_action, agent, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps / 1000)}k, Episode Reward:{ep_r}')
                'save model'
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps / 1000))
        env.close()
        eval_env.close()

def Action_adapter_reverse(act, max_action):
    return act / max_action


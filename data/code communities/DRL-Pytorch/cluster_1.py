# Cluster 1

def evaluate_policy(env, agent, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = dw or tr
            total_scores += r
            s = s_next
    return int(total_scores / turns)

def main():
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    env = gym.make(EnvName[opt.EnvIdex])
    eval_env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    if opt.DDQN:
        algo_name = 'DDQN'
    else:
        algo_name = 'DQN'
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    print('Algorithm:', algo_name, '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')
    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2:]
        writepath = 'runs/Prior{}_{}'.format(algo_name, BriefEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    model = DQN_Agent(opt)
    if opt.Loadmodel:
        model.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)
    buffer = PrioritizedReplayBuffer(opt)
    exp_noise_scheduler = LinearSchedule(opt.noise_decay_steps, opt.exp_noise_init, opt.exp_noise_end)
    beta_scheduler = LinearSchedule(opt.beta_gain_steps, opt.beta_init, 1.0)
    if opt.render:
        score = evaluate_policy(eval_env, model, 5)
        print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset()
            done, ep_step = (False, 0)
            while not done:
                ep_step += 1
                if buffer.size < opt.warmup:
                    a = env.action_space.sample()
                else:
                    a = model.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a)
                if r <= -100:
                    r = -10
                buffer.add(s, a, r, s_next, dw)
                done = dw + tr
                s = s_next
                model.exp_noise = exp_noise_scheduler.value(total_steps)
                buffer.beta = beta_scheduler.value(total_steps)
                'update if its time'
                if total_steps >= opt.warmup and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(buffer)
                'record & log'
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, model)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('p_sum', buffer.sum_tree.priority_sum, global_step=total_steps)
                        writer.add_scalar('p_max', buffer.sum_tree.priority_max, global_step=total_steps)
                        writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                        writer.add_scalar('beta', buffer.beta, global_step=total_steps)
                    print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
                total_steps += 1
                'save model'
                if total_steps % opt.save_interval == 0:
                    model.save(algo_name, BriefEnvName[opt.EnvIdex], total_steps)
    env.close()

def main():
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    Env_With_DW = [True, True]
    opt.env_with_dw = Env_With_DW[opt.EnvIdex]
    env = gym.make(EnvName[opt.EnvIdex])
    eval_env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    if opt.DDQN:
        algo_name = 'DDQN'
    else:
        algo_name = 'DQN'
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    print('Algorithm:', algo_name, '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')
    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2:]
        writepath = 'runs/LightPrior{}_{}'.format(algo_name, BriefEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    model = DQN_Agent(opt)
    if opt.Loadmodel:
        model.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)
    buffer = LightPriorReplayBuffer(opt)
    exp_noise_scheduler = LinearSchedule(opt.noise_decay_steps, opt.exp_noise_init, opt.exp_noise_end)
    beta_scheduler = LinearSchedule(opt.beta_gain_steps, opt.beta_init, 1.0)
    lr_scheduler = LinearSchedule(opt.lr_decay_steps, opt.lr_init, opt.lr_end)
    if opt.render:
        score = evaluate_policy(eval_env, model, 20)
        print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset()
            a, q_a = model.select_action(s, deterministic=False)
            while True:
                s_next, r, dw, tr, info = env.step(a)
                if r <= -100:
                    r = -10
                a_next, q_a_next = model.select_action(s_next, deterministic=False)
                priority = (torch.abs(r + ~dw * opt.gamma * q_a_next - q_a) + 0.01) ** opt.alpha
                buffer.add(s, a, r, dw, tr, priority)
                s, a, q_a = (s_next, a_next, q_a_next)
                'update if its time'
                if total_steps >= opt.warmup and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(buffer)
                    model.exp_noise = exp_noise_scheduler.value(total_steps)
                    buffer.beta = beta_scheduler.value(total_steps)
                    for p in model.q_net_optimizer.param_groups:
                        p['lr'] = lr_scheduler.value(total_steps)
                'record & log'
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, model)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                        writer.add_scalar('beta', buffer.beta, global_step=total_steps)
                    print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
                total_steps += 1
                'save model'
                if total_steps % opt.save_interval == 0:
                    model.save(algo_name, BriefEnvName[opt.EnvIdex], int(total_steps / 1000))
                if dw or tr:
                    break
    env.close()
    eval_env.close()

def main():
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    env = gym.make(EnvName[opt.EnvIdex])
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    if opt.DDQN:
        algo_name = 'DDQN'
    else:
        algo_name = 'DQN'
    torch.manual_seed(opt.seed)
    env.seed(opt.seed)
    env.action_space.seed(opt.seed)
    eval_env.seed(opt.seed)
    eval_env.action_space.seed(opt.seed)
    np.random.seed(opt.seed)
    print('Algorithm:', algo_name, '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')
    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2:]
        writepath = 'runs/Prior{}_{}'.format(algo_name, BriefEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    model = DQN_Agent(opt)
    if opt.Loadmodel:
        model.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)
    buffer = PrioritizedReplayBuffer(opt)
    exp_noise_scheduler = LinearSchedule(opt.noise_decay_steps, opt.exp_noise_init, opt.exp_noise_end)
    beta_scheduler = LinearSchedule(opt.beta_gain_steps, opt.beta_init, 1.0)
    if opt.render:
        score = evaluate_policy(eval_env, model, True, 20)
        print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, done, ep_r, steps = (env.reset(), False, 0, 0)
            while not done:
                steps += 1
                if buffer.size < opt.warmup:
                    a = env.action_space.sample()
                else:
                    a = model.select_action(s, deterministic=False)
                s_next, r, done, info = env.step(a)
                'Avoid impacts caused by reaching max episode steps'
                if done and steps != opt.max_e_steps:
                    dw = True
                    if opt.EnvIdex == 1:
                        if r <= -100:
                            r = -10
                else:
                    dw = False
                buffer.add(s, a, r, s_next, dw)
                s = s_next
                ep_r += r
                model.exp_noise = exp_noise_scheduler.value(total_steps)
                buffer.beta = beta_scheduler.value(total_steps)
                'update if its time'
                if total_steps >= opt.warmup and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(buffer)
                'record & log'
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, model, render=False)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('p_sum', buffer.sum_tree.priority_sum, global_step=total_steps)
                        writer.add_scalar('p_max', buffer.sum_tree.priority_max, global_step=total_steps)
                        writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                        writer.add_scalar('beta', buffer.beta, global_step=total_steps)
                    print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
                total_steps += 1
                'save model'
                if total_steps % opt.save_interval == 0:
                    model.save(algo_name, BriefEnvName[opt.EnvIdex], total_steps)
    env.close()

def main():
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed: {}'.format(opt.seed))
    print('Algorithm: SACD', '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2:]
        writepath = 'runs/SACD_{}'.format(BriefEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = SACD_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(opt.ModelIdex, BriefEnvName[opt.EnvIdex])
    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
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
                if opt.EnvIdex == 1:
                    if r <= -100:
                        r = -10
                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                'update if its time'
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        agent.train()
                'record & log'
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('alpha', agent.alpha, global_step=total_steps)
                        writer.add_scalar('H_mean', agent.H_mean, global_step=total_steps)
                    print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
                total_steps += 1
                'save model'
                if total_steps % opt.save_interval == 0:
                    agent.save(int(total_steps / 1000), BriefEnvName[opt.EnvIdex])
    env.close()
    eval_env.close()

def main():
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    if opt.Duel:
        algo_name = 'Duel'
    else:
        algo_name = ''
    if opt.Double:
        algo_name += 'DDQN'
    else:
        algo_name += 'DQN'
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed: {}'.format(opt.seed))
    print('Algorithm:', algo_name, '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2:]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name, BriefEnvName[opt.EnvIdex], opt.seed) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)
    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
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
                'Update'
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        agent.train()
                'Noise decay & Record & Log'
                if total_steps % 1000 == 0:
                    agent.exp_noise *= opt.noise_decay
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                    print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
                total_steps += 1
                'save model'
                if total_steps % opt.save_interval == 0:
                    agent.save(algo_name, BriefEnvName[opt.EnvIdex], int(total_steps / 1000))
    env.close()
    eval_env.close()

def main():
    write = True
    Loadmodel = False
    Max_train_steps = 20000
    seed = 0
    np.random.seed(seed)
    print(f'Random Seed: {seed}')
    ' ↓↓↓ Build Env ↓↓↓ '
    EnvName = 'CliffWalking-v0'
    env = gym.make(EnvName)
    env = TimeLimit(env, max_episode_steps=500)
    eval_env = gym.make(EnvName)
    eval_env = TimeLimit(eval_env, max_episode_steps=100)
    ' ↓↓↓ Use tensorboard to record training curves ↓↓↓ '
    if write:
        timenow = str(datetime.now())[0:-7]
        timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2:]
        writepath = 'runs/{}'.format(EnvName) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    ' ↓↓↓ Build Q-learning Agent ↓↓↓ '
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = QLearningAgent(s_dim=env.observation_space.n, a_dim=env.action_space.n, lr=0.2, gamma=0.9, exp_noise=0.1)
    if Loadmodel:
        agent.restore()
    ' ↓↓↓ Iterate and Train ↓↓↓ '
    total_steps = 0
    while total_steps < Max_train_steps:
        s, info = env.reset(seed=seed)
        seed += 1
        done, steps = (False, 0)
        while not done:
            steps += 1
            a = agent.select_action(s, deterministic=False)
            s_next, r, dw, tr, info = env.step(a)
            agent.train(s, a, r, s_next, dw)
            done = dw or tr
            s = s_next
            total_steps += 1
            'record & log'
            if total_steps % 100 == 0:
                ep_r = evaluate_policy(eval_env, agent)
                if write:
                    writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                print(f'EnvName:{EnvName}, Seed:{seed}, Steps:{total_steps}, Episode reward:{ep_r}')
            'save model'
            if total_steps % Max_train_steps == 0:
                agent.save()
    env.close()
    eval_env.close()

def main():
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    algo_name = 'NoisyNetDQN'
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed: {}'.format(opt.seed))
    print('Algorithm:', algo_name, '  Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2:]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name, BriefEnvName[opt.EnvIdex], opt.seed) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = NoisyNetDQN_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)
    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
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
                    a = agent.select_action(s)
                s_next, r, dw, tr, info = env.step(a)
                done = dw or tr
                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                'Update'
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        agent.train()
                'Record & Log'
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
                total_steps += 1
                'Save model'
                if total_steps % opt.save_interval == 0:
                    agent.save(algo_name, BriefEnvName[opt.EnvIdex], int(total_steps / 1000))
    env.close()
    eval_env.close()

def main():
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CP-v1', 'LLd-v2']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed: {}'.format(opt.seed))
    print('Env:', BriefEnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '   Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps)
    print('\n')
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2:]
        writepath = 'runs/{}'.format(BriefEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = PPO_discrete(**vars(opt))
    if opt.Loadmodel:
        agent.load(opt.ModelIdex)
    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, turns=1)
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
                s_next, r, dw, tr, info = env.step(a)
                if r <= -100:
                    r = -30
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
                    score = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:', EnvName[opt.EnvIdex], 'seed:', opt.seed, 'steps: {}k'.format(int(total_steps / 1000)), 'score:', score)
                'Save model'
                if total_steps % opt.save_interval == 0:
                    agent.save(total_steps)
        env.close()
        eval_env.close()


# Cluster 6

def main():
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    env = gym.make(EnvName[opt.EnvIdex], render_mode='human' if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    opt.action_info = {0: ['Left', 'Right'], 1: ['Noop', 'LeftEngine', 'MainEngine', 'RightEngine']}
    algo_name = 'C51_' + 'DDQN' if opt.DQL else 'DQN'
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
        writepath = 'runs/{}_{}'.format(algo_name, BriefEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    if not os.path.exists('model'):
        os.mkdir('model')
    agent = CDQN_agent(**vars(opt))
    if opt.Loadmodel:
        agent.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)
    if opt.render:
        render_policy(env, agent, opt)
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
                'update if its time'
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        agent.train()
                'record & log'
                if total_steps % opt.eval_interval == 0:
                    agent.exp_noise *= opt.noise_decay
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

def render_policy(env, agent, opt):
    plt.ion()
    x_range = torch.linspace(opt.v_min, opt.v_max, steps=opt.n_atoms).numpy()
    width = (opt.v_max - opt.v_min) / (opt.n_atoms - 1)
    while True:
        s, info = env.reset()
        done, Episodic_scores = (False, 0)
        with torch.no_grad():
            while not done:
                s = torch.FloatTensor(s.reshape(1, -1)).to(opt.dvc)
                distributions, q_values = agent.q_net._predict(s)
                a = torch.argmax(q_values, dim=1).cpu().item()
                s_next, r, dw, tr, info = env.step(a)
                done = dw or tr
                Episodic_scores += r
                s = s_next
                dists = distributions.squeeze().cpu().numpy()
                for i in range(opt.action_dim):
                    plt.bar(x_range, dists[i], width=width, label=opt.action_info[opt.EnvIdex][i])
                plt.ylim(0, 0.6)
                plt.legend(loc='upper left')
                plt.title('C51 by XinJingHao')
                plt.pause(1e-05)
                plt.clf()
        print('Episodic scores:', int(Episodic_scores))


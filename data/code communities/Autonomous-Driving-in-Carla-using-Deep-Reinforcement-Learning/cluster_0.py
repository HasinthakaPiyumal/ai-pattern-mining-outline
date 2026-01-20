# Cluster 0

def runner():
    args = parse_args()
    exp_name = args.exp_name
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init
    try:
        if exp_name == 'ppo':
            run_name = 'PPO'
        else:
            '\n            \n            Here the functionality can be extended to different algorithms.\n\n            '
            sys.exit()
    except Exception as e:
        print(e.message)
        sys.exit()
    if train == True:
        writer = SummaryWriter(f'runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}')
    else:
        writer = SummaryWriter(f'runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}')
    writer.add_text('hyperparameters', '|param|value|\n|-|-|\n%s' % '\n'.join([f'|{key}|{value}' for key, value in vars(args).items()]))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    action_std_decay_rate = 0.05
    min_action_std = 0.05
    action_std_decay_freq = 500000.0
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0
    try:
        client, world = ClientConnection(town).setup()
        logging.info('Connection has been setup successfully.')
    except:
        logging.error('Connection has been refused by the server.')
        ConnectionRefusedError
    if train:
        env = CarlaEnvironment(client, world, town)
    else:
        env = CarlaEnvironment(client, world, town, checkpoint_frequency=None)
    encode = EncodeState(LATENT_DIM)
    try:
        time.sleep(0.5)
        if checkpoint_load:
            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1
            chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_' + str(chkt_file_nums) + '.pickle'
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']
                action_std_init = data['action_std_init']
            agent = PPOAgent(town, action_std_init)
            agent.load()
        elif train == False:
            agent = PPOAgent(town, action_std_init)
            agent.load()
            for params in agent.old_policy.actor.parameters():
                params.requires_grad = False
        else:
            agent = PPOAgent(town, action_std_init)
        if train:
            while timestep < total_timesteps:
                observation = env.reset()
                observation = encode.process(observation)
                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(args.episode_length):
                    action = agent.get_action(observation, train=True)
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encode.process(observation)
                    agent.memory.rewards.append(reward)
                    agent.memory.dones.append(done)
                    timestep += 1
                    current_ep_reward += reward
                    if timestep % action_std_decay_freq == 0:
                        action_std_init = agent.decay_action_std(action_std_decay_rate, min_action_std)
                    if timestep == total_timesteps - 1:
                        agent.chkpt_save()
                    if done:
                        episode += 1
                        t2 = datetime.now()
                        t3 = t2 - t1
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info[1]
                distance_covered += info[0]
                scores.append(current_ep_reward)
                if checkpoint_load:
                    cumulative_score = (cumulative_score * (episode - 1) + current_ep_reward) / episode
                else:
                    cumulative_score = np.mean(scores)
                print('Episode: {}'.format(episode), ', Timestep: {}'.format(timestep), ', Reward:  {:.2f}'.format(current_ep_reward), ', Average Reward:  {:.2f}'.format(cumulative_score))
                if episode % 10 == 0:
                    agent.learn()
                    agent.chkpt_save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    if chkt_file_nums != 0:
                        chkt_file_nums -= 1
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_' + str(chkt_file_nums) + '.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                if episode % 5 == 0:
                    writer.add_scalar('Episodic Reward/episode', scores[-1], episode)
                    writer.add_scalar('Cumulative Reward/info', cumulative_score, episode)
                    writer.add_scalar('Cumulative Reward/(t)', cumulative_score, timestep)
                    writer.add_scalar('Average Episodic Reward/info', np.mean(scores[-5]), episode)
                    writer.add_scalar('Average Reward/(t)', np.mean(scores[-5]), timestep)
                    writer.add_scalar('Episode Length (s)/info', np.mean(episodic_length), episode)
                    writer.add_scalar('Reward/(t)', current_ep_reward, timestep)
                    writer.add_scalar('Average Deviation from Center/episode', deviation_from_center / 5, episode)
                    writer.add_scalar('Average Deviation from Center/(t)', deviation_from_center / 5, timestep)
                    writer.add_scalar('Average Distance Covered (m)/episode', distance_covered / 5, episode)
                    writer.add_scalar('Average Distance Covered (m)/(t)', distance_covered / 5, timestep)
                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0
                if episode % 100 == 0:
                    agent.save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_' + str(chkt_file_nums) + '.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
            print('Terminating the run.')
            sys.exit()
        else:
            while timestep < args.test_timesteps:
                observation = env.reset()
                observation = encode.process(observation)
                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(args.episode_length):
                    action = agent.get_action(observation, train=False)
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encode.process(observation)
                    timestep += 1
                    current_ep_reward += reward
                    if done:
                        episode += 1
                        t2 = datetime.now()
                        t3 = t2 - t1
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info[1]
                distance_covered += info[0]
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)
                print('Episode: {}'.format(episode), ', Timestep: {}'.format(timestep), ', Reward:  {:.2f}'.format(current_ep_reward), ', Average Reward:  {:.2f}'.format(cumulative_score))
                writer.add_scalar('TEST: Episodic Reward/episode', scores[-1], episode)
                writer.add_scalar('TEST: Cumulative Reward/info', cumulative_score, episode)
                writer.add_scalar('TEST: Cumulative Reward/(t)', cumulative_score, timestep)
                writer.add_scalar('TEST: Episode Length (s)/info', np.mean(episodic_length), episode)
                writer.add_scalar('TEST: Reward/(t)', current_ep_reward, timestep)
                writer.add_scalar('TEST: Deviation from Center/episode', deviation_from_center, episode)
                writer.add_scalar('TEST: Deviation from Center/(t)', deviation_from_center, timestep)
                writer.add_scalar('TEST: Distance Covered (m)/episode', distance_covered, episode)
                writer.add_scalar('TEST: Distance Covered (m)/(t)', distance_covered, timestep)
                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0
            print('Terminating the run.')
            sys.exit()
    finally:
        sys.exit()


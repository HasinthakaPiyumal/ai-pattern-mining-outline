# Cluster 30

class Learner:

    def __init__(self, env_args, train_args, eval_args, policy_args, seed, **kwargs):
        self.seed = seed
        self.init_env(**env_args)
        self.init_agent(**policy_args)
        self.init_train(**train_args)
        self.init_eval(**eval_args)

    def init_env(self, env_type, env_name, max_rollouts_per_task=None, num_tasks=None, num_train_tasks=None, num_eval_tasks=None, eval_envs=None, worst_percentile=None, **kwargs):
        assert env_type in ['meta', 'pomdp', 'credit', 'rmdp', 'generalize', 'atari']
        self.env_type = env_type
        if self.env_type == 'meta':
            from envs.meta.make_env import make_env
            self.train_env = make_env(env_name, max_rollouts_per_task, seed=self.seed, n_tasks=num_tasks, **kwargs)
            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)
            if self.train_env.n_tasks is not None:
                assert num_train_tasks >= num_eval_tasks > 0
                shuffled_tasks = np.random.permutation(self.train_env.unwrapped.get_all_task_idx())
                self.train_tasks = shuffled_tasks[:num_train_tasks]
                self.eval_tasks = shuffled_tasks[-num_eval_tasks:]
            else:
                assert num_tasks == num_train_tasks == None
                assert num_eval_tasks > 0
                self.train_tasks = []
                self.eval_tasks = num_eval_tasks * [None]
            self.max_rollouts_per_task = max_rollouts_per_task
            self.max_trajectory_len = self.train_env.horizon_bamdp
        elif self.env_type in ['pomdp', 'credit']:
            import envs.pomdp
            import envs.credit_assign
            assert num_eval_tasks > 0
            self.train_env = gym.make(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)
            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)
            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps
        elif self.env_type == 'atari':
            from envs.atari import create_env
            assert num_eval_tasks > 0
            self.train_env = create_env(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)
            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)
            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps
        elif self.env_type == 'rmdp':
            sys.path.append('envs/rl-generalization')
            import sunblaze_envs
            assert num_eval_tasks > 0 and worst_percentile > 0.0 and (worst_percentile < 1.0)
            self.train_env = sunblaze_envs.make(env_name, **kwargs)
            self.train_env.seed(self.seed)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)
            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)
            self.worst_percentile = worst_percentile
            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps
        elif self.env_type == 'generalize':
            sys.path.append('envs/rl-generalization')
            import sunblaze_envs
            self.train_env = sunblaze_envs.make(env_name, **kwargs)
            self.train_env.seed(self.seed)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)

            def check_env_class(env_name):
                if 'Normal' in env_name:
                    return 'R'
                if 'Extreme' in env_name:
                    return 'E'
                return 'D'
            self.train_env_name = check_env_class(env_name)
            self.eval_envs = {}
            for env_name, num_eval_task in eval_envs.items():
                eval_env = sunblaze_envs.make(env_name, **kwargs)
                eval_env.seed(self.seed + 1)
                self.eval_envs[eval_env] = (check_env_class(env_name), num_eval_task)
            logger.log(self.train_env_name, self.train_env)
            logger.log(self.eval_envs)
            self.train_tasks = []
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps
        else:
            raise ValueError
        if self.train_env.action_space.__class__.__name__ == 'Box':
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == 'Discrete'
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        self.obs_dim = self.train_env.observation_space.shape[0]
        logger.log('obs_dim', self.obs_dim, 'act_dim', self.act_dim)

    def init_agent(self, seq_model, separate: bool=True, image_encoder=None, reward_clip=False, **kwargs):
        if seq_model == 'mlp':
            agent_class = AGENT_CLASSES['Policy_MLP']
            rnn_encoder_type = None
            assert separate == True
        elif '-mlp' in seq_model:
            agent_class = AGENT_CLASSES['Policy_RNN_MLP']
            rnn_encoder_type = seq_model.split('-')[0]
            assert separate == True
        else:
            rnn_encoder_type = seq_model
            if separate == True:
                agent_class = AGENT_CLASSES['Policy_Separate_RNN']
            else:
                agent_class = AGENT_CLASSES['Policy_Shared_RNN']
        self.agent_arch = agent_class.ARCH
        logger.log(agent_class, self.agent_arch)
        if image_encoder is not None:
            image_encoder_fn = lambda: ImageEncoder(image_shape=self.train_env.image_space.shape, **image_encoder)
        else:
            image_encoder_fn = lambda: None
        self.agent = agent_class(encoder=rnn_encoder_type, obs_dim=self.obs_dim, action_dim=self.act_dim, image_encoder_fn=image_encoder_fn, **kwargs).to(ptu.device)
        logger.log(self.agent)
        self.reward_clip = reward_clip

    def init_train(self, buffer_size, batch_size, num_iters, num_init_rollouts_pool, num_rollouts_per_iter, num_updates_per_iter=None, sampled_seq_len=None, sample_weight_baseline=None, buffer_type=None, **kwargs):
        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(num_updates_per_iter, float)
        self.num_updates_per_iter = num_updates_per_iter
        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleReplayBuffer(max_replay_buffer_size=int(buffer_size), observation_dim=self.obs_dim, action_dim=self.act_dim if self.act_continuous else 1, max_trajectory_len=self.max_trajectory_len, add_timeout=False)
        else:
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len
            if buffer_type is None or buffer_type == SeqReplayBuffer.buffer_type:
                buffer_class = SeqReplayBuffer
            elif buffer_type == RAMEfficient_SeqReplayBuffer.buffer_type:
                buffer_class = RAMEfficient_SeqReplayBuffer
            logger.log(buffer_class)
            self.policy_storage = buffer_class(max_replay_buffer_size=int(buffer_size), observation_dim=self.obs_dim, action_dim=self.act_dim if self.act_continuous else 1, sampled_seq_len=sampled_seq_len, sample_weight_baseline=sample_weight_baseline, observation_type=self.train_env.observation_space.dtype)
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter
        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log('*** total rollouts', total_rollouts, 'total env steps', self.n_env_steps_total)

    def init_eval(self, log_interval, save_interval, log_tensorboard, eval_stochastic=False, num_episodes_per_task=1, **kwargs):
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard
        self.eval_stochastic = eval_stochastic
        self.eval_num_episodes_per_task = num_episodes_per_task

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0
        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self):
        """
        training loop
        """
        self._start_training()
        if self.num_init_rollouts_pool > 0:
            logger.log('Collecting initial pool of data..')
            while self._n_env_steps_total < self.num_init_rollouts_pool * self.max_trajectory_len:
                self.collect_rollouts(num_rollouts=1, random_actions=True)
            logger.log('Done! env steps', self._n_env_steps_total, 'rollouts', self._n_rollouts_total)
            if isinstance(self.num_updates_per_iter, float):
                train_stats = self.update(int(self._n_env_steps_total * self.num_updates_per_iter))
                self.log_train_stats(train_stats)
        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)
            logger.log('env steps', self._n_env_steps_total)
            train_stats = self.update(self.num_updates_per_iter if isinstance(self.num_updates_per_iter, int) else int(math.ceil(self.num_updates_per_iter * env_steps)))
            self.log_train_stats(train_stats)
            current_num_iters = self._n_env_steps_total // (self.num_rollouts_per_iter * self.max_trajectory_len)
            if current_num_iters != last_eval_num_iters and current_num_iters % self.log_interval == 0:
                last_eval_num_iters = current_num_iters
                perf = self.log()
                if self.save_interval > 0 and self._n_env_steps_total > 0.75 * self.n_env_steps_total and (current_num_iters % self.save_interval == 0):
                    self.save_model(current_num_iters, perf)
        self.save_model(current_num_iters, perf)

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """
        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0
            if self.env_type == 'meta' and self.train_env.n_tasks is not None:
                task = self.train_tasks[np.random.randint(len(self.train_tasks))]
                obs = ptu.from_numpy(self.train_env.reset(task=task))
            else:
                obs = ptu.from_numpy(self.train_env.reset())
            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False
            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                obs_list, act_list, rew_list, next_obs_list, term_list = ([], [], [], [], [])
            if self.agent_arch == AGENT_ARCHS.Memory:
                action, reward, internal_state = self.agent.get_initial_info()
            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor([self.train_env.action_space.sample()])
                    if not self.act_continuous:
                        action = F.one_hot(action.long(), num_classes=self.act_dim).float()
                elif self.agent_arch == AGENT_ARCHS.Memory:
                    (action, _, _, _), internal_state = self.agent.act(prev_internal_state=internal_state, prev_action=action, reward=reward, obs=obs, deterministic=False)
                else:
                    action, _, _, _ = self.agent.act(obs, deterministic=False)
                next_obs, reward, done, info = utl.env_step(self.train_env, action.squeeze(dim=0))
                if self.reward_clip and self.env_type == 'atari':
                    reward = torch.tanh(reward)
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                steps += 1
                if self.env_type == 'meta' and 'is_goal_state' in dir(self.train_env.unwrapped):
                    term = self.train_env.unwrapped.is_goal_state()
                    self._successes_in_buffer += int(term)
                elif self.env_type == 'credit':
                    term = done_rollout
                else:
                    term = False if 'TimeLimit.truncated' in info or steps >= self.max_trajectory_len else done_rollout
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(observation=ptu.get_numpy(obs.squeeze(dim=0)), action=ptu.get_numpy(action.squeeze(dim=0) if self.act_continuous else torch.argmax(action.squeeze(dim=0), dim=-1, keepdims=True)), reward=ptu.get_numpy(reward.squeeze(dim=0)), terminal=np.array([term], dtype=float), next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)))
                else:
                    obs_list.append(obs)
                    act_list.append(action)
                    rew_list.append(reward)
                    term_list.append(term)
                    next_obs_list.append(next_obs)
                obs = next_obs.clone()
            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                act_buffer = torch.cat(act_list, dim=0)
                if not self.act_continuous:
                    act_buffer = torch.argmax(act_buffer, dim=-1, keepdims=True)
                self.policy_storage.add_episode(observations=ptu.get_numpy(torch.cat(obs_list, dim=0)), actions=ptu.get_numpy(act_buffer), rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)), terminals=np.array(term_list).reshape(-1, 1), next_observations=ptu.get_numpy(torch.cat(next_obs_list, dim=0)))
                print(f'steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}')
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:
            batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            batch = self.sample_rl_batch(self.batch_size)
            rl_losses = self.agent.update(batch)
            for k, v in rl_losses.items():
                if update == 0:
                    rl_losses_agg[k] = [v]
                else:
                    rl_losses_agg[k].append(v)
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates
        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True):
        num_episodes = self.max_rollouts_per_task
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))
        if self.env_type == 'meta':
            num_steps_per_episode = self.eval_env.unwrapped._max_episode_steps
            obs_size = self.eval_env.unwrapped.observation_space.shape[0]
            observations = np.zeros((len(tasks), self.max_trajectory_len + 1, obs_size))
        else:
            num_steps_per_episode = self.eval_env._max_episode_steps
            observations = None
        for task_idx, task in enumerate(tasks):
            step = 0
            if self.env_type == 'meta' and self.eval_env.n_tasks is not None:
                obs = ptu.from_numpy(self.eval_env.reset(task=task))
                observations[task_idx, step, :] = ptu.get_numpy(obs[:obs_size])
            else:
                obs = ptu.from_numpy(self.eval_env.reset())
            obs = obs.reshape(1, obs.shape[-1])
            if self.agent_arch == AGENT_ARCHS.Memory:
                action, reward, internal_state = self.agent.get_initial_info()
            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for _ in range(num_steps_per_episode):
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(prev_internal_state=internal_state, prev_action=action, reward=reward, obs=obs, deterministic=deterministic)
                    else:
                        action, _, _, _ = self.agent.act(obs, deterministic=deterministic)
                    next_obs, reward, done, info = utl.env_step(self.eval_env, action.squeeze(dim=0))
                    running_reward += reward.item()
                    if self.reward_clip and self.env_type == 'atari':
                        reward = torch.tanh(reward)
                    step += 1
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                    if self.env_type == 'meta':
                        observations[task_idx, step, :] = ptu.get_numpy(next_obs[0, :obs_size])
                    obs = next_obs.clone()
                    if self.env_type == 'meta' and 'is_goal_state' in dir(self.eval_env.unwrapped) and self.eval_env.unwrapped.is_goal_state():
                        success_rate[task_idx] = 1.0
                    elif self.env_type == 'generalize' and self.eval_env.unwrapped.is_success():
                        success_rate[task_idx] = 1.0
                    elif 'success' in info and info['success'] == True:
                        success_rate[task_idx] = 1.0
                    if done_rollout:
                        break
                    if self.env_type == 'meta' and info['done_mdp'] == True:
                        break
                returns_per_episode[task_idx, episode_idx] = running_reward
            total_steps[task_idx] = step
        return (returns_per_episode, success_rate, observations, total_steps)

    def log_train_stats(self, train_stats):
        logger.record_step(self._n_env_steps_total)
        for k, v in train_stats.items():
            logger.record_tabular('rl_loss/' + k, v)
        if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                logger.record_tabular('rl_loss/' + k, v)
        logger.dump_tabular()

    def log(self):
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular('z/env_steps', self._n_env_steps_total)
        logger.record_tabular('z/rollouts', self._n_rollouts_total)
        logger.record_tabular('z/rl_steps', self._n_rl_update_steps_total)
        if self.env_type == 'meta':
            if self.train_env.n_tasks is not None:
                returns_train, success_rate_train, observations, total_steps_train = self.evaluate(self.train_tasks[:len(self.eval_tasks)])
            returns_eval, success_rate_eval, observations_eval, total_steps_eval = self.evaluate(self.eval_tasks)
            if self.eval_stochastic:
                returns_eval_sto, success_rate_eval_sto, observations_eval_sto, total_steps_eval_sto = self.evaluate(self.eval_tasks, deterministic=False)
            if self.train_env.n_tasks is not None and 'plot_behavior' in dir(self.eval_env.unwrapped):
                for i, task in enumerate(self.train_tasks[:min(5, len(self.eval_tasks))]):
                    self.eval_env.reset(task=task)
                    logger.add_figure('trajectory/train_task_{}'.format(i), utl_eval.plot_rollouts(observations[i, :], self.eval_env))
                for i, task in enumerate(self.eval_tasks[:min(5, len(self.eval_tasks))]):
                    self.eval_env.reset(task=task)
                    logger.add_figure('trajectory/eval_task_{}'.format(i), utl_eval.plot_rollouts(observations_eval[i, :], self.eval_env))
                    if self.eval_stochastic:
                        logger.add_figure('trajectory/eval_task_{}_sto'.format(i), utl_eval.plot_rollouts(observations_eval_sto[i, :], self.eval_env))
            if 'is_goal_state' in dir(self.eval_env.unwrapped):
                logger.record_tabular('metrics/successes_in_buffer', self._successes_in_buffer / self._n_env_steps_total)
                if self.train_env.n_tasks is not None:
                    logger.record_tabular('metrics/success_rate_train', np.mean(success_rate_train))
                logger.record_tabular('metrics/success_rate_eval', np.mean(success_rate_eval))
                if self.eval_stochastic:
                    logger.record_tabular('metrics/success_rate_eval_sto', np.mean(success_rate_eval_sto))
            for episode_idx in range(self.max_rollouts_per_task):
                if self.train_env.n_tasks is not None:
                    logger.record_tabular('metrics/return_train_episode_{}'.format(episode_idx + 1), np.mean(returns_train[:, episode_idx]))
                logger.record_tabular('metrics/return_eval_episode_{}'.format(episode_idx + 1), np.mean(returns_eval[:, episode_idx]))
                if self.eval_stochastic:
                    logger.record_tabular('metrics/return_eval_episode_{}_sto'.format(episode_idx + 1), np.mean(returns_eval_sto[:, episode_idx]))
            if self.train_env.n_tasks is not None:
                logger.record_tabular('metrics/total_steps_train', np.mean(total_steps_train))
                logger.record_tabular('metrics/return_train_total', np.mean(np.sum(returns_train, axis=-1)))
            logger.record_tabular('metrics/total_steps_eval', np.mean(total_steps_eval))
            logger.record_tabular('metrics/return_eval_total', np.mean(np.sum(returns_eval, axis=-1)))
            if self.eval_stochastic:
                logger.record_tabular('metrics/total_steps_eval_sto', np.mean(total_steps_eval_sto))
                logger.record_tabular('metrics/return_eval_total_sto', np.mean(np.sum(returns_eval_sto, axis=-1)))
        elif self.env_type == 'generalize':
            returns_eval, success_rate_eval, total_steps_eval = ({}, {}, {})
            for env, (env_name, eval_num_episodes_per_task) in self.eval_envs.items():
                self.eval_env = env
                for suffix, deterministic in zip(['', '_sto'], [True, False]):
                    if deterministic == False and self.eval_stochastic == False:
                        continue
                    return_eval, success_eval, _, total_step_eval = self.evaluate(eval_num_episodes_per_task * [None], deterministic=deterministic)
                    returns_eval[self.train_env_name + env_name + suffix] = return_eval.squeeze(-1)
                    success_rate_eval[self.train_env_name + env_name + suffix] = success_eval
                    total_steps_eval[self.train_env_name + env_name + suffix] = total_step_eval
            for k, v in returns_eval.items():
                logger.record_tabular(f'metrics/return_eval_{k}', np.mean(v))
            for k, v in success_rate_eval.items():
                logger.record_tabular(f'metrics/succ_eval_{k}', np.mean(v))
            for k, v in total_steps_eval.items():
                logger.record_tabular(f'metrics/total_steps_eval_{k}', np.mean(v))
        elif self.env_type == 'rmdp':
            returns_eval, _, _, total_steps_eval = self.evaluate(self.eval_tasks)
            returns_eval = returns_eval.squeeze(-1)
            cutoff = np.percentile(returns_eval, 100 * self.worst_percentile)
            worst_indices = np.where(returns_eval <= cutoff)
            returns_eval_worst, total_steps_eval_worst = (returns_eval[worst_indices], total_steps_eval[worst_indices])
            logger.record_tabular('metrics/return_eval_avg', returns_eval.mean())
            logger.record_tabular('metrics/return_eval_worst', returns_eval_worst.mean())
            logger.record_tabular('metrics/total_steps_eval_avg', total_steps_eval.mean())
            logger.record_tabular('metrics/total_steps_eval_worst', total_steps_eval_worst.mean())
        elif self.env_type in ['pomdp', 'credit', 'atari']:
            returns_eval, success_rate_eval, _, total_steps_eval = self.evaluate(self.eval_tasks)
            if self.eval_stochastic:
                returns_eval_sto, success_rate_eval_sto, _, total_steps_eval_sto = self.evaluate(self.eval_tasks, deterministic=False)
            logger.record_tabular('metrics/total_steps_eval', np.mean(total_steps_eval))
            logger.record_tabular('metrics/return_eval_total', np.mean(np.sum(returns_eval, axis=-1)))
            logger.record_tabular('metrics/success_rate_eval', np.mean(success_rate_eval))
            if self.eval_stochastic:
                logger.record_tabular('metrics/total_steps_eval_sto', np.mean(total_steps_eval_sto))
                logger.record_tabular('metrics/return_eval_total_sto', np.mean(np.sum(returns_eval_sto, axis=-1)))
                logger.record_tabular('metrics/success_rate_eval_sto', np.mean(success_rate_eval_sto))
        else:
            raise ValueError
        logger.record_tabular('z/time_cost', int(time.time() - self._start_time))
        logger.record_tabular('z/fps', (self._n_env_steps_total - self._n_env_steps_total_last) / (time.time() - self._start_time_last))
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()
        logger.dump_tabular()
        if self.env_type == 'generalize':
            return sum([v.mean() for v in success_rate_eval.values()]) / len(success_rate_eval)
        else:
            return np.mean(np.sum(returns_eval, axis=-1))

    def save_model(self, iter, perf):
        save_path = os.path.join(logger.get_dir(), 'save', f'agent_{iter}_perf{perf:.3f}.pt')
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        print('load successfully from', ckpt_path)


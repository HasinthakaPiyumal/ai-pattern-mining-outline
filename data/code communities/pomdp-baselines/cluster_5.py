# Cluster 5

def get_augmented_obs(args, obs, posterior_sample=None, task_mu=None, task_std=None):
    obs_augmented = obs.clone()
    if posterior_sample is None:
        sample_embeddings = False
    else:
        sample_embeddings = args.sample_embeddings
    if not args.condition_policy_on_state:
        obs_augmented = ptu.zeros(0)
    if sample_embeddings and posterior_sample is not None:
        obs_augmented = torch.cat((obs_augmented, posterior_sample), dim=1)
    elif task_mu is not None and task_std is not None:
        task_mu = task_mu.reshape((-1, task_mu.shape[-1]))
        task_std = task_std.reshape((-1, task_std.shape[-1]))
        obs_augmented = torch.cat((obs_augmented, task_mu, task_std), dim=-1)
    return obs_augmented

def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)

class FeatureExtractor(nn.Module):
    """one-layer MLP with relu
    Used for extracting features for vector-based observations/actions/rewards

    NOTE: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    torch.linear is a linear transformation in the LAST dimension
    with weight of size (IN, OUT)
    which means it can support the input size larger than 2-dim, in the form
    of (*, IN), and then transform into (*, OUT) with same size (*)
    e.g. In the encoder, the input is (N, B, IN) where N=seq_len.
    """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return ptu.zeros(0)

class TanhNormal(Distribution):
    """aka squashed normal
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-06):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return (torch.tanh(z), z)
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.atanh(value)
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample()
        if return_pretanh_value:
            return (torch.tanh(z), z)
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = self.normal_mean + self.normal_std * Variable(Normal(ptu.zeros(self.normal_mean.size()), ptu.ones(self.normal_std.size())).sample())
        if return_pretanh_value:
            return (torch.tanh(z), z)
        else:
            return torch.tanh(z)

def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)

class ModelFreeOffPolicy_Shared_RNN(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with shared RNN
    """
    ARCH = 'memory'

    def __init__(self, obs_dim, action_dim, encoder, algo_name, action_embedding_size, observ_embedding_size, reward_embedding_size, rnn_hidden_size, dqn_layers, policy_layers, lr=0.0003, gamma=0.99, tau=0.005, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.algo = RL_ALGORITHMS[algo_name](**kwargs[algo_name], action_dim=action_dim)
        self.observ_embedder = utl.FeatureExtractor(obs_dim, observ_embedding_size, F.relu)
        self.action_embedder = utl.FeatureExtractor(action_dim, action_embedding_size, F.relu)
        self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)
        rnn_input_size = action_embedding_size + observ_embedding_size + reward_embedding_size
        self.rnn_hidden_size = rnn_hidden_size
        assert encoder in [LSTM_name, GRU_name]
        self.encoder = encoder
        self.num_layers = 1
        self.rnn = RNNs[encoder](input_size=rnn_input_size, hidden_size=self.rnn_hidden_size, num_layers=self.num_layers, batch_first=False, bias=True)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.current_observ_action_embedder = utl.FeatureExtractor(obs_dim + action_dim, rnn_input_size, F.relu)
        self.qf1, self.qf2 = self.algo.build_critic(input_size=self.rnn_hidden_size + rnn_input_size, hidden_sizes=dqn_layers, action_dim=action_dim)
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)
        self.current_observ_embedder = utl.FeatureExtractor(obs_dim, observ_embedding_size, F.relu)
        self.policy = self.algo.build_actor(input_size=self.rnn_hidden_size + observ_embedding_size, action_dim=self.action_dim, hidden_sizes=policy_layers)
        self.policy_target = deepcopy(self.policy)
        self.optimizer = Adam([*self.observ_embedder.parameters(), *self.action_embedder.parameters(), *self.reward_embedder.parameters(), *self.rnn.parameters(), *self.current_observ_action_embedder.parameters(), *self.current_observ_embedder.parameters(), *self.qf1.parameters(), *self.qf2.parameters(), *self.policy.parameters()], lr=lr)

    def get_hidden_states(self, prev_actions, rewards, observs, initial_internal_state=None):
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self.observ_embedder(observs)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)
        if initial_internal_state is None:
            output, _ = self.rnn(inputs)
            return output
        else:
            output, current_internal_state = self.rnn(inputs, initial_internal_state)
            return (output, current_internal_state)

    def forward(self, actions, rewards, observs, dones, masks):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert actions.dim() == rewards.dim() == dones.dim() == observs.dim() == masks.dim() == 3
        assert actions.shape[0] == rewards.shape[0] == dones.shape[0] == observs.shape[0] == masks.shape[0] + 1
        num_valid = torch.clamp(masks.sum(), min=1.0)
        hidden_states = self.get_hidden_states(prev_actions=actions, rewards=rewards, observs=observs)
        obs_embeds = self.current_observ_embedder(observs)
        joint_policy_embeds = torch.cat((hidden_states, obs_embeds), dim=-1)
        with torch.no_grad():
            new_next_actions, new_next_log_probs = self.algo.forward_actor_in_target(actor=self.policy, actor_target=self.policy_target, next_observ=joint_policy_embeds)
            obs_act_embeds = self.current_observ_action_embedder(torch.cat((observs, new_next_actions), dim=-1))
            joint_q_embeds = torch.cat((hidden_states, obs_act_embeds), dim=-1)
            next_q1 = self.qf1_target(joint_q_embeds)
            next_q2 = self.qf2_target(joint_q_embeds)
            min_next_q_target = torch.min(next_q1, next_q2)
            min_next_q_target += self.algo.entropy_bonus(new_next_log_probs)
            q_target = rewards + (1.0 - dones) * self.gamma * min_next_q_target
            q_target = q_target[1:]
        curr_obs_act_embeds = self.current_observ_action_embedder(torch.cat((observs[:-1], actions[1:]), dim=-1))
        curr_joint_q_embeds = torch.cat((hidden_states[:-1], curr_obs_act_embeds), dim=-1)
        q1_pred = self.qf1(curr_joint_q_embeds)
        q2_pred = self.qf2(curr_joint_q_embeds)
        q1_pred, q2_pred = (q1_pred * masks, q2_pred * masks)
        q_target = q_target * masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid
        new_actions, new_log_probs = self.algo.forward_actor(actor=self.policy, observ=joint_policy_embeds)
        new_obs_act_embeds = self.current_observ_action_embedder(torch.cat((observs, new_actions), dim=-1))
        new_joint_q_embeds = torch.cat((hidden_states, new_obs_act_embeds), dim=-1)
        q1 = self.qf1(new_joint_q_embeds)
        q2 = self.qf2(new_joint_q_embeds)
        min_q_new_actions = torch.min(q1, q2)
        policy_loss = -min_q_new_actions
        policy_loss += -self.algo.entropy_bonus(new_log_probs)
        policy_loss = policy_loss[:-1]
        policy_loss = (policy_loss * masks).sum() / num_valid
        total_loss = 0.5 * (qf1_loss + qf2_loss) + policy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        outputs = {'qf1_loss': qf1_loss.item(), 'qf2_loss': qf2_loss.item(), 'policy_loss': policy_loss.item()}
        self.soft_target_update()
        if new_log_probs is not None:
            with torch.no_grad():
                current_log_probs = (new_log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()
            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)
        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def report_grad_norm(self):
        return {'rnn_grad_norm': utl.get_grad_norm(self.rnn), 'q_grad_norm': utl.get_grad_norm(self.qf1), 'pi_grad_norm': utl.get_grad_norm(self.policy)}

    def update(self, batch):
        actions, rewards, dones = (batch['act'], batch['rew'], batch['term'])
        _, batch_size, _ = actions.shape
        masks = batch['mask']
        obs, next_obs = (batch['obs'], batch['obs2'])
        observs = torch.cat((obs[[0]], next_obs), dim=0)
        actions = torch.cat((ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0)
        rewards = torch.cat((ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0)
        dones = torch.cat((ptu.zeros((1, batch_size, 1)).float(), dones), dim=0)
        return self.forward(actions, rewards, observs, dones, masks)

    @torch.no_grad()
    def get_initial_info(self):
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
        hidden_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
        if self.encoder == GRU_name:
            internal_state = hidden_state
        else:
            cell_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
            internal_state = (hidden_state, cell_state)
        return (prev_action, reward, internal_state)

    @torch.no_grad()
    def act(self, prev_internal_state, prev_action, reward, obs, deterministic=False, return_log_prob=False):
        prev_action = prev_action.unsqueeze(0)
        reward = reward.unsqueeze(0)
        obs = obs.unsqueeze(0)
        hidden_state, current_internal_state = self.get_hidden_states(prev_actions=prev_action, rewards=reward, observs=obs, initial_internal_state=prev_internal_state)
        curr_embed = self.current_observ_embedder(obs)
        joint_embeds = torch.cat((hidden_state, curr_embed), dim=-1)
        if joint_embeds.dim() == 3:
            joint_embeds = joint_embeds.squeeze(0)
        action_tuple = self.algo.select_action(actor=self.policy, observ=joint_embeds, deterministic=deterministic, return_log_prob=return_log_prob)
        return (action_tuple, current_internal_state)

class ModelFreeOffPolicy_Separate_RNN(nn.Module):
    """Recommended Architecture
    Recurrent Actor and Recurrent Critic with separate RNNs
    """
    ARCH = 'memory'
    Markov_Actor = False
    Markov_Critic = False

    def __init__(self, obs_dim, action_dim, encoder, algo_name, action_embedding_size, observ_embedding_size, reward_embedding_size, rnn_hidden_size, dqn_layers, policy_layers, rnn_num_layers=1, lr=0.0003, gamma=0.99, tau=0.005, image_encoder_fn=lambda: None, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.algo = RL_ALGORITHMS[algo_name](**kwargs[algo_name], action_dim=action_dim)
        self.critic = Critic_RNN(obs_dim, action_dim, encoder, self.algo, action_embedding_size, observ_embedding_size, reward_embedding_size, rnn_hidden_size, dqn_layers, rnn_num_layers, image_encoder=image_encoder_fn())
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = deepcopy(self.critic)
        self.actor = Actor_RNN(obs_dim, action_dim, encoder, self.algo, action_embedding_size, observ_embedding_size, reward_embedding_size, rnn_hidden_size, policy_layers, rnn_num_layers, image_encoder=image_encoder_fn())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.actor_target = deepcopy(self.actor)

    @torch.no_grad()
    def get_initial_info(self):
        return self.actor.get_initial_info()

    @torch.no_grad()
    def act(self, prev_internal_state, prev_action, reward, obs, deterministic=False, return_log_prob=False):
        prev_action = prev_action.unsqueeze(0)
        reward = reward.unsqueeze(0)
        obs = obs.unsqueeze(0)
        current_action_tuple, current_internal_state = self.actor.act(prev_internal_state=prev_internal_state, prev_action=prev_action, reward=reward, obs=obs, deterministic=deterministic, return_log_prob=return_log_prob)
        return (current_action_tuple, current_internal_state)

    def forward(self, actions, rewards, observs, dones, masks):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert actions.dim() == rewards.dim() == dones.dim() == observs.dim() == masks.dim() == 3
        assert actions.shape[0] == rewards.shape[0] == dones.shape[0] == observs.shape[0] == masks.shape[0] + 1
        num_valid = torch.clamp(masks.sum(), min=1.0)
        (q1_pred, q2_pred), q_target = self.algo.critic_loss(markov_actor=self.Markov_Actor, markov_critic=self.Markov_Critic, actor=self.actor, actor_target=self.actor_target, critic=self.critic, critic_target=self.critic_target, observs=observs, actions=actions, rewards=rewards, dones=dones, gamma=self.gamma)
        q1_pred, q2_pred = (q1_pred * masks, q2_pred * masks)
        q_target = q_target * masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid
        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()
        policy_loss, log_probs = self.algo.actor_loss(markov_actor=self.Markov_Actor, markov_critic=self.Markov_Critic, actor=self.actor, actor_target=self.actor_target, critic=self.critic, critic_target=self.critic_target, observs=observs, actions=actions, rewards=rewards)
        policy_loss = (policy_loss * masks).sum() / num_valid
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        outputs = {'qf1_loss': qf1_loss.item(), 'qf2_loss': qf2_loss.item(), 'policy_loss': policy_loss.item()}
        self.soft_target_update()
        if log_probs is not None:
            with torch.no_grad():
                current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()
            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)
        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        return {'q_grad_norm': utl.get_grad_norm(self.critic), 'q_rnn_grad_norm': utl.get_grad_norm(self.critic.rnn), 'pi_grad_norm': utl.get_grad_norm(self.actor), 'pi_rnn_grad_norm': utl.get_grad_norm(self.actor.rnn)}

    def update(self, batch):
        actions, rewards, dones = (batch['act'], batch['rew'], batch['term'])
        _, batch_size, _ = actions.shape
        if not self.algo.continuous_action:
            actions = F.one_hot(actions.squeeze(-1).long(), num_classes=self.action_dim).float()
        masks = batch['mask']
        obs, next_obs = (batch['obs'], batch['obs2'])
        observs = torch.cat((obs[[0]], next_obs), dim=0)
        actions = torch.cat((ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0)
        rewards = torch.cat((ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0)
        dones = torch.cat((ptu.zeros((1, batch_size, 1)).float(), dones), dim=0)
        return self.forward(actions, rewards, observs, dones, masks)

class Actor_RNN(nn.Module):

    def __init__(self, obs_dim, action_dim, encoder, algo, action_embedding_size, observ_embedding_size, reward_embedding_size, rnn_hidden_size, policy_layers, rnn_num_layers, image_encoder=None, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo
        self.image_encoder = image_encoder
        if self.image_encoder is None:
            self.observ_embedder = utl.FeatureExtractor(obs_dim, observ_embedding_size, F.relu)
        else:
            assert observ_embedding_size == 0
            observ_embedding_size = self.image_encoder.embed_size
        self.action_embedder = utl.FeatureExtractor(action_dim, action_embedding_size, F.relu)
        self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)
        rnn_input_size = action_embedding_size + observ_embedding_size + reward_embedding_size
        self.rnn_hidden_size = rnn_hidden_size
        assert encoder in RNNs
        self.encoder = encoder
        self.num_layers = rnn_num_layers
        self.rnn = RNNs[encoder](input_size=rnn_input_size, hidden_size=self.rnn_hidden_size, num_layers=self.num_layers, batch_first=False, bias=True)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        if self.image_encoder is None:
            self.current_observ_embedder = utl.FeatureExtractor(obs_dim, observ_embedding_size, F.relu)
        self.policy = self.algo.build_actor(input_size=self.rnn_hidden_size + observ_embedding_size, action_dim=self.action_dim, hidden_sizes=policy_layers)

    def _get_obs_embedding(self, observs):
        if self.image_encoder is None:
            return self.observ_embedder(observs)
        else:
            return self.image_encoder(observs)

    def _get_shortcut_obs_embedding(self, observs):
        if self.image_encoder is None:
            return self.current_observ_embedder(observs)
        else:
            return self.image_encoder(observs)

    def get_hidden_states(self, prev_actions, rewards, observs, initial_internal_state=None):
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self._get_obs_embedding(observs)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)
        if initial_internal_state is None:
            output, _ = self.rnn(inputs)
            return output
        else:
            output, current_internal_state = self.rnn(inputs, initial_internal_state)
            return (output, current_internal_state)

    def forward(self, prev_actions, rewards, observs):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        return current actions a' (T+1, B, dim) based on previous history

        """
        assert prev_actions.dim() == rewards.dim() == observs.dim() == 3
        assert prev_actions.shape[0] == rewards.shape[0] == observs.shape[0]
        hidden_states = self.get_hidden_states(prev_actions=prev_actions, rewards=rewards, observs=observs)
        curr_embed = self._get_shortcut_obs_embedding(observs)
        joint_embeds = torch.cat((hidden_states, curr_embed), dim=-1)
        return self.algo.forward_actor(actor=self.policy, observ=joint_embeds)

    @torch.no_grad()
    def get_initial_info(self):
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
        hidden_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
        if self.encoder == GRU_name:
            internal_state = hidden_state
        else:
            cell_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
            internal_state = (hidden_state, cell_state)
        return (prev_action, reward, internal_state)

    @torch.no_grad()
    def act(self, prev_internal_state, prev_action, reward, obs, deterministic=False, return_log_prob=False):
        hidden_state, current_internal_state = self.get_hidden_states(prev_actions=prev_action, rewards=reward, observs=obs, initial_internal_state=prev_internal_state)
        curr_embed = self._get_shortcut_obs_embedding(obs)
        joint_embeds = torch.cat((hidden_state, curr_embed), dim=-1)
        if joint_embeds.dim() == 3:
            joint_embeds = joint_embeds.squeeze(0)
        action_tuple = self.algo.select_action(actor=self.policy, observ=joint_embeds, deterministic=deterministic, return_log_prob=return_log_prob)
        return (action_tuple, current_internal_state)


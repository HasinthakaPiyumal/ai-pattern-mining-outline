# Cluster 32

class TD3(RLAlgorithmBase):
    name = 'td3'
    continuous_action = True
    use_target_actor = True

    def __init__(self, exploration_noise=0.1, target_noise=0.2, target_noise_clip=0.5, **kwargs):
        self.exploration_noise = exploration_noise
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return DeterministicPolicy(obs_dim=input_size, action_dim=action_dim, hidden_sizes=hidden_sizes, **kwargs)

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(input_size=input_size, output_size=1, hidden_sizes=hidden_sizes)
        qf2 = FlattenMlp(input_size=input_size, output_size=1, hidden_sizes=hidden_sizes)
        return (qf1, qf2)

    def select_action(self, actor, observ, deterministic: bool, **kwargs):
        mean = actor(observ)
        if deterministic:
            action_tuple = (mean, mean, None, None)
        else:
            action = (mean + torch.randn_like(mean) * self.exploration_noise).clamp(-1, 1)
            action_tuple = (action, mean, None, None)
        return action_tuple

    @staticmethod
    def forward_actor(actor, observ):
        new_actions = actor(observ)
        return (new_actions, None)

    def _inject_noise(self, actions):
        action_noise = (torch.randn_like(actions) * self.target_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
        new_actions = (actions + action_noise).clamp(-1, 1)
        return new_actions

    def critic_loss(self, markov_actor: bool, markov_critic: bool, actor, actor_target, critic, critic_target, observs, actions, rewards, dones, gamma, next_observs=None):
        with torch.no_grad():
            if markov_actor:
                new_actions, _ = self.forward_actor(actor_target, next_observs if markov_critic else observs)
            else:
                new_actions, _ = actor_target(prev_actions=actions, rewards=rewards, observs=next_observs if markov_critic else observs)
            new_actions = self._inject_noise(new_actions)
            if markov_critic:
                next_q1 = critic_target[0](next_observs, new_actions)
                next_q2 = critic_target[1](next_observs, new_actions)
            else:
                next_q1, next_q2 = critic_target(prev_actions=actions, rewards=rewards, observs=observs, current_actions=new_actions)
            min_next_q_target = torch.min(next_q1, next_q2)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target
            if not markov_critic:
                q_target = q_target[1:]
        if markov_critic:
            q1_pred = critic[0](observs, actions)
            q2_pred = critic[1](observs, actions)
        else:
            q1_pred, q2_pred = critic(prev_actions=actions, rewards=rewards, observs=observs, current_actions=actions[1:])
        return ((q1_pred, q2_pred), q_target)

    def actor_loss(self, markov_actor: bool, markov_critic: bool, actor, actor_target, critic, critic_target, observs, actions=None, rewards=None):
        if markov_actor:
            new_actions, _ = self.forward_actor(actor, observs)
        else:
            new_actions, _ = actor(prev_actions=actions, rewards=rewards, observs=observs)
        if markov_critic:
            q1 = critic[0](observs, new_actions)
            q2 = critic[1](observs, new_actions)
        else:
            q1, q2 = critic(prev_actions=actions, rewards=rewards, observs=observs, current_actions=new_actions)
        min_q_new_actions = torch.min(q1, q2)
        policy_loss = -min_q_new_actions
        if not markov_critic:
            policy_loss = policy_loss[:-1]
        return (policy_loss, None)

    def forward_actor_in_target(self, actor, actor_target, next_observ):
        new_next_actions, _ = self.forward_actor(actor_target, next_observ)
        return (self._inject_noise(new_next_actions), None)

    def entropy_bonus(self, log_probs):
        return 0.0

class SACD(RLAlgorithmBase):
    name = 'sacd'
    continuous_action = False
    use_target_actor = False

    def __init__(self, entropy_alpha=0.1, automatic_entropy_tuning=True, target_entropy=None, alpha_lr=0.0003, action_dim=None):
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            assert target_entropy is not None
            self.target_entropy = float(target_entropy) * np.log(action_dim)
            self.log_alpha_entropy = torch.zeros(1, requires_grad=True, device=ptu.device)
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = entropy_alpha

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return CategoricalPolicy(obs_dim=input_size, action_dim=action_dim, hidden_sizes=hidden_sizes, **kwargs)

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        assert action_dim is not None
        if obs_dim is not None:
            input_size = obs_dim
        qf1 = FlattenMlp(input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes)
        qf2 = FlattenMlp(input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes)
        return (qf1, qf2)

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool):
        action, prob, log_prob = actor(observ, deterministic, return_log_prob)
        return (action, prob, log_prob, None)

    @staticmethod
    def forward_actor(actor, observ):
        _, probs, log_probs = actor(observ, return_log_prob=True)
        return (probs, log_probs)

    def critic_loss(self, markov_actor: bool, markov_critic: bool, actor, actor_target, critic, critic_target, observs, actions, rewards, dones, gamma, next_observs=None):
        with torch.no_grad():
            if markov_actor:
                new_probs, new_log_probs = self.forward_actor(actor, next_observs if markov_critic else observs)
            else:
                new_probs, new_log_probs = actor(prev_actions=actions, rewards=rewards, observs=next_observs if markov_critic else observs)
            if markov_critic:
                next_q1 = critic_target[0](next_observs)
                next_q2 = critic_target[1](next_observs)
            else:
                next_q1, next_q2 = critic_target(prev_actions=actions, rewards=rewards, observs=observs, current_actions=new_probs)
            min_next_q_target = torch.min(next_q1, next_q2)
            min_next_q_target += self.alpha_entropy * -new_log_probs
            min_next_q_target = (new_probs * min_next_q_target).sum(dim=-1, keepdims=True)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target
            if not markov_critic:
                q_target = q_target[1:]
        if markov_critic:
            q1_pred = critic[0](observs)
            q2_pred = critic[1](observs)
            action = actions.long()
            q1_pred = q1_pred.gather(dim=-1, index=action)
            q2_pred = q2_pred.gather(dim=-1, index=action)
        else:
            q1_pred, q2_pred = critic(prev_actions=actions, rewards=rewards, observs=observs, current_actions=actions[1:])
            stored_actions = actions[1:]
            stored_actions = torch.argmax(stored_actions, dim=-1, keepdims=True)
            q1_pred = q1_pred.gather(dim=-1, index=stored_actions)
            q2_pred = q2_pred.gather(dim=-1, index=stored_actions)
        return ((q1_pred, q2_pred), q_target)

    def actor_loss(self, markov_actor: bool, markov_critic: bool, actor, actor_target, critic, critic_target, observs, actions=None, rewards=None):
        if markov_actor:
            new_probs, log_probs = self.forward_actor(actor, observs)
        else:
            new_probs, log_probs = actor(prev_actions=actions, rewards=rewards, observs=observs)
        if markov_critic:
            q1 = critic[0](observs)
            q2 = critic[1](observs)
        else:
            q1, q2 = critic(prev_actions=actions, rewards=rewards, observs=observs, current_actions=new_probs)
        min_q_new_actions = torch.min(q1, q2)
        policy_loss = -min_q_new_actions
        policy_loss += self.alpha_entropy * log_probs
        policy_loss = (new_probs * policy_loss).sum(axis=-1, keepdims=True)
        if not markov_critic:
            policy_loss = policy_loss[:-1]
        log_probs = (new_probs * log_probs).sum(axis=-1, keepdims=True)
        return (policy_loss, log_probs)

    def update_others(self, current_log_probs):
        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (current_log_probs + self.target_entropy)
            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().item()
        return {'policy_entropy': -current_log_probs, 'alpha': self.alpha_entropy}

class SAC(RLAlgorithmBase):
    name = 'sac'
    continuous_action = True
    use_target_actor = False

    def __init__(self, entropy_alpha=0.1, automatic_entropy_tuning=True, target_entropy=None, alpha_lr=0.0003, action_dim=None):
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = float(target_entropy)
            else:
                self.target_entropy = -float(action_dim)
            self.log_alpha_entropy = torch.zeros(1, requires_grad=True, device=ptu.device)
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = entropy_alpha

    def update_others(self, current_log_probs):
        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (current_log_probs + self.target_entropy)
            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().item()
        return {'policy_entropy': -current_log_probs, 'alpha': self.alpha_entropy}

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return TanhGaussianPolicy(obs_dim=input_size, action_dim=action_dim, hidden_sizes=hidden_sizes, **kwargs)

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(input_size=input_size, output_size=1, hidden_sizes=hidden_sizes)
        qf2 = FlattenMlp(input_size=input_size, output_size=1, hidden_sizes=hidden_sizes)
        return (qf1, qf2)

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool):
        return actor(observ, False, deterministic, return_log_prob)

    @staticmethod
    def forward_actor(actor, observ):
        new_actions, _, _, log_probs = actor(observ, return_log_prob=True)
        return (new_actions, log_probs)

    def critic_loss(self, markov_actor: bool, markov_critic: bool, actor, actor_target, critic, critic_target, observs, actions, rewards, dones, gamma, next_observs=None):
        with torch.no_grad():
            if markov_actor:
                new_actions, new_log_probs = self.forward_actor(actor, next_observs if markov_critic else observs)
            else:
                new_actions, new_log_probs = actor(prev_actions=actions, rewards=rewards, observs=next_observs if markov_critic else observs)
            if markov_critic:
                next_q1 = critic_target[0](next_observs, new_actions)
                next_q2 = critic_target[1](next_observs, new_actions)
            else:
                next_q1, next_q2 = critic_target(prev_actions=actions, rewards=rewards, observs=observs, current_actions=new_actions)
            min_next_q_target = torch.min(next_q1, next_q2)
            min_next_q_target += self.alpha_entropy * -new_log_probs
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target
            if not markov_critic:
                q_target = q_target[1:]
        if markov_critic:
            q1_pred = critic[0](observs, actions)
            q2_pred = critic[1](observs, actions)
        else:
            q1_pred, q2_pred = critic(prev_actions=actions, rewards=rewards, observs=observs, current_actions=actions[1:])
        return ((q1_pred, q2_pred), q_target)

    def actor_loss(self, markov_actor: bool, markov_critic: bool, actor, actor_target, critic, critic_target, observs, actions=None, rewards=None):
        if markov_actor:
            new_actions, log_probs = self.forward_actor(actor, observs)
        else:
            new_actions, log_probs = actor(prev_actions=actions, rewards=rewards, observs=observs)
        if markov_critic:
            q1 = critic[0](observs, new_actions)
            q2 = critic[1](observs, new_actions)
        else:
            q1, q2 = critic(prev_actions=actions, rewards=rewards, observs=observs, current_actions=new_actions)
        min_q_new_actions = torch.min(q1, q2)
        policy_loss = -min_q_new_actions
        policy_loss += self.alpha_entropy * log_probs
        if not markov_critic:
            policy_loss = policy_loss[:-1]
        return (policy_loss, log_probs)

    def forward_actor_in_target(self, actor, actor_target, next_observ):
        return self.forward_actor(actor, next_observ)

    def entropy_bonus(self, log_probs):
        return self.alpha_entropy * -log_probs


# Cluster 14

class Agent:

    def __init__(self, cfg):
        self.whole_cfg = cfg
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num
        self.game_player_id = self.whole_cfg.agent.game_player_id
        self.game_team_id = self.game_player_id // self.player_num
        self.player_id = self.whole_cfg.agent.player_id
        self.features = Features(self.whole_cfg)
        self.eval_padding = self.whole_cfg.agent.get('eval_padding', False)
        self.use_action_mask = self.whole_cfg.agent.get('use_action_mask', False)
        self.model = Model(self.whole_cfg)

    def reset(self):
        self.last_action_type = self.features.direction_num * 2

    def preprocess(self, obs):
        self.last_player_score = obs[1][self.game_player_id]['score']
        if self.use_action_mask:
            can_eject = obs[1][self.game_player_id]['can_eject']
            can_split = obs[1][self.game_player_id]['can_split']
            action_mask = self.features.generate_action_mask(can_eject=can_eject, can_split=can_split)
        else:
            action_mask = self.features.generate_action_mask(can_eject=True, can_split=True)
        obs = self.features.transform_obs(obs, game_player_id=self.game_player_id, last_action_type=self.last_action_type, padding=self.eval_padding)
        obs = default_collate_with_dim([obs])
        obs['action_mask'] = action_mask.unsqueeze(0)
        return obs

    def step(self, obs):
        self.raw_obs = obs
        obs = self.preprocess(obs)
        self.model_input = obs
        with torch.no_grad():
            self.model_output = self.model.compute_action(self.model_input)
        actions = self.postprocess(self.model_output['action'].detach().numpy())
        return actions

    def postprocess(self, model_actions):
        actions = {}
        actions[self.game_player_id] = self.features.transform_action(model_actions[0])
        self.last_action_type = model_actions[0].item()
        return actions

class Agent:

    def __init__(self, cfg=None):
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.agent
        self.use_action_mask = self.whole_cfg.agent.get('use_action_mask', False)
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num
        self.game_player_id = self.whole_cfg.agent.game_player_id
        self.game_team_id = self.game_player_id // self.player_num
        self.features = Features(self.whole_cfg)
        self.device = 'cpu'
        self.model = Model(self.whole_cfg)

    def transform_action(self, agent_outputs, env_status, eval_vsbot=False):
        env_num = len(env_status)
        actions_list = agent_outputs['action'].cpu().numpy().tolist()
        actions = {}
        for env_id in range(env_num):
            actions[env_id] = {}
            game_player_num = self.player_num if eval_vsbot else self.player_num * self.team_num
            for game_player_id in range(game_player_num):
                action_idx = actions_list[env_id * game_player_num + game_player_id]
                env_status[env_id].last_action_types[game_player_id] = action_idx
                actions[env_id][game_player_id] = self.features.transform_action(action_idx)
        return actions

    def reset(self):
        self.last_action_type = {}
        for player_id in range(self.player_num * self.game_team_id, self.player_num * (self.game_team_id + 1)):
            self.last_action_type[player_id] = self.features.direction_num * 2

    def step(self, obs):
        """
        Overview:
            Agent.step() in submission
        Arguments:
            - obs
        Returns:
            - action
        """
        env_team_obs = []
        for player_id in range(self.player_num * self.game_team_id, self.player_num * (self.game_team_id + 1)):
            game_player_obs = self.features.transform_obs(obs, game_player_id=player_id, last_action_type=self.last_action_type[player_id])
            env_team_obs.append(game_player_obs)
        env_team_obs = stack(env_team_obs)
        obs = default_collate_with_dim([env_team_obs], device=self.device)
        self.model_input = obs
        with torch.no_grad():
            model_output = self.model(self.model_input)['action'].cpu().detach().numpy()
        actions = []
        for i in range(len(model_output)):
            actions.append(self.features.transform_action(model_output[i]))
        ret = {}
        for player_id, act in zip(range(self.player_num * self.game_team_id, self.player_num * (self.game_team_id + 1)), actions):
            ret[player_id] = act
        for player_id, act in zip(range(self.player_num * self.game_team_id, self.player_num * (self.game_team_id + 1)), model_output):
            self.last_action_type[player_id] = act.item()
        return ret


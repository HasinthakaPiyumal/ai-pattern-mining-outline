# Cluster 22

def env_step(env, action):
    action = ptu.get_numpy(action)
    if env.action_space.__class__.__name__ == 'Discrete':
        action = np.argmax(action)
    next_obs, reward, done, info = env.step(action)
    next_obs = ptu.from_numpy(next_obs).view(-1, next_obs.shape[0])
    reward = ptu.FloatTensor([reward]).view(-1, 1)
    done = ptu.from_numpy(np.array(done, dtype=int)).view(-1, 1)
    return (next_obs, reward, done, info)

def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)

@torch.no_grad()
def collect_rollouts(num_rollouts, random_actions=False, deterministic=False, train_mode=True):
    """collect num_rollouts of trajectories in task and save into policy buffer
    :param
        random_actions: whether to use policy to sample actions, or randomly sample action space
        deterministic: deterministic action selection?
        train_mode: whether to train (stored to buffer) or test
    """
    if not train_mode:
        assert random_actions == False and deterministic == True
    total_steps = 0
    total_rewards = 0.0
    for idx in range(num_rollouts):
        steps = 0
        rewards = 0.0
        obs = ptu.from_numpy(env.reset())
        obs = obs.reshape(1, obs.shape[-1])
        done_rollout = False
        action, reward, internal_state = agent.get_initial_info()
        if train_mode:
            obs_list, act_list, rew_list, next_obs_list, term_list = ([], [], [], [], [])
        while not done_rollout:
            if random_actions:
                action = ptu.FloatTensor([env.action_space.sample()])
            else:
                (action, _, _, _), internal_state = agent.act(prev_internal_state=internal_state, prev_action=action, reward=reward, obs=obs, deterministic=deterministic)
            next_obs, reward, done, info = utl.env_step(env, action.squeeze(dim=0))
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
            steps += 1
            rewards += reward.item()
            term = False if 'TimeLimit.truncated' in info or steps >= max_trajectory_len else done_rollout
            if train_mode:
                obs_list.append(obs)
                act_list.append(action)
                rew_list.append(reward)
                term_list.append(term)
                next_obs_list.append(next_obs)
            obs = next_obs.clone()
        if train_mode:
            policy_storage.add_episode(observations=ptu.get_numpy(torch.cat(obs_list, dim=0)), actions=ptu.get_numpy(torch.cat(act_list, dim=0)), rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)), terminals=np.array(term_list).reshape(-1, 1), next_observations=ptu.get_numpy(torch.cat(next_obs_list, dim=0)))
        print('Mode:', 'Train' if train_mode else 'Test', 'env_steps', steps, 'total rewards', rewards)
        total_steps += steps
        total_rewards += rewards
    if train_mode:
        return total_steps
    else:
        return total_rewards / num_rollouts


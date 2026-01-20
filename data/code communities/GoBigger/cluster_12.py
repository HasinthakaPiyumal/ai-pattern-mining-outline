# Cluster 12

def create_env(env_name, custom_cfg={}, **kwargs):
    """
    env_name choice in ['st_v0', 'sp_v0']
    """
    cfg = importlib.import_module('gobigger.configs.{}'.format(env_name))
    cfg = eval('cfg.{}'.format(env_name))
    cfg = deep_merge_dicts(cfg, custom_cfg)
    if env_name.startswith('st'):
        return create_env_st(cfg, **kwargs)
    elif env_name.startswith('sp'):
        return create_env_sp(cfg, **kwargs)
    else:
        raise NotImplementedError

def create_env_st(cfg, **kwargs):
    return GoBiggerEnv(cfg, **kwargs)

def create_env_sp(cfg, **kwargs):
    return GoBiggerSPEnv(cfg, **kwargs)

def create_env_custom(type, cfg=None, **kwargs):
    if type == 'st':
        return create_env_st(cfg, **kwargs)
    elif type == 'sp':
        return create_env_sp(cfg, **kwargs)
    else:
        raise NotImplementedError

@pytest.mark.unittest
class TestPlayback:

    def test_none_pb(self):
        env = create_env('st_t2p2', dict(frame_limit=100, playback_settings=dict(playback_type='none')))
        obs = env.reset()
        bot_agents = []
        team_infos = env.get_team_infos()
        print(team_infos)
        for team_id, player_ids in team_infos:
            for player_id in player_ids:
                bot_agents.append(BotAgent(player_id, level=2))
        time_step_all = 0
        for i in range(100000):
            actions = {bot_agent.name: bot_agent.step(obs[1][bot_agent.name]) for bot_agent in bot_agents}
            t1 = time.time()
            obs, reward, done, info = env.step(actions=actions)
            t2 = time.time()
            time_step_all += t2 - t1
            logging.debug('{} {:.4f} envstep {:.3f} / {:.3f}, leaderboard={}'.format(i, obs[0]['last_frame_count'], t2 - t1, time_step_all / (i + 1), obs[0]['leaderboard']))
            if done:
                logging.debug('Game Over')
                break
        env.close()

    def test_video_pb(self):
        env = create_env('st_t2p2', dict(frame_limit=100, playback_settings=dict(playback_type='by_video', by_video=dict(save_video=True))))
        obs = env.reset()
        bot_agents = []
        team_infos = env.get_team_infos()
        print(team_infos)
        for team_id, player_ids in team_infos:
            for player_id in player_ids:
                bot_agents.append(BotAgent(player_id, level=2))
        time_step_all = 0
        for i in range(100000):
            actions = {bot_agent.name: bot_agent.step(obs[1][bot_agent.name]) for bot_agent in bot_agents}
            t1 = time.time()
            obs, reward, done, info = env.step(actions=actions)
            t2 = time.time()
            time_step_all += t2 - t1
            logging.debug('{} {:.4f} envstep {:.3f} / {:.3f}, leaderboard={}'.format(i, obs[0]['last_frame_count'], t2 - t1, time_step_all / (i + 1), obs[0]['leaderboard']))
            if done:
                logging.debug('Game Over')
                break
        env.close()
        assert os.path.isfile('test-all.mp4')
        os.remove('test-all.mp4')

    def test_frame_pb(self):
        env = create_env('st_t2p2', dict(frame_limit=100, playback_settings=dict(playback_type='by_frame', by_frame=dict(save_frame=True))))
        obs = env.reset()
        bot_agents = []
        team_infos = env.get_team_infos()
        print(team_infos)
        for team_id, player_ids in team_infos:
            for player_id in player_ids:
                bot_agents.append(BotAgent(player_id, level=2))
        time_step_all = 0
        for i in range(100000):
            actions = {bot_agent.name: bot_agent.step(obs[1][bot_agent.name]) for bot_agent in bot_agents}
            t1 = time.time()
            obs, reward, done, info = env.step(actions=actions)
            t2 = time.time()
            time_step_all += t2 - t1
            logging.debug('{} {:.4f} envstep {:.3f} / {:.3f}, leaderboard={}'.format(i, obs[0]['last_frame_count'], t2 - t1, time_step_all / (i + 1), obs[0]['leaderboard']))
            if done:
                logging.debug('Game Over')
                break
        env.close()
        assert os.path.isfile('test.pb')
        os.remove('test.pb')


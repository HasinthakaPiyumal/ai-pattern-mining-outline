# Cluster 3

def test_api():
    env = gym.make('sumo-rl-v0', num_seconds=100, use_gui=False, net_file='sumo_rl/nets/single-intersection/single-intersection.net.xml', route_file='sumo_rl/nets/single-intersection/single-intersection.rou.xml')
    env.reset()
    check_env(env.unwrapped, skip_render_check=True)
    env.close()

def test_parallel_api():
    env = sumo_rl.parallel_env(net_file='sumo_rl/nets/4x4-Lucas/4x4.net.xml', route_file='sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml', out_csv_name='outputs/4x4grid/test', use_gui=False, num_seconds=100)
    parallel_api_test(env, num_cycles=10)
    env.close()


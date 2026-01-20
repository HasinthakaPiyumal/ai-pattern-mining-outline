# Cluster 0

def main():
    description = 'CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n'
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost', help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000', help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0', help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='', help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default='60.0', help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--routes', help='Name of the route to be executed. Point to the route_xml_file to be executed.', required=True)
    parser.add_argument('--scenarios', help='Name of the scenario annotation file to be mixed with the route.', required=True)
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions per route.')
    parser.add_argument('-a', '--agent', type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument('--agent-config', type=str, help="Path to Agent's configuration file", default='')
    parser.add_argument('--track', type=str, default='SENSORS', help='Participation track: SENSORS, MAP')
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument('--checkpoint', type=str, default='./simulation_results.json', help='Path to checkpoint used for saving statistics and resuming')
    arguments = parser.parse_args()
    statistics_manager = StatisticsManager()
    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)
    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


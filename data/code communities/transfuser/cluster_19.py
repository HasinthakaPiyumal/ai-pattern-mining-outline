# Cluster 19

def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    args = argparser.parse_args()
    args.rolename = 'hero'
    args.filter = 'vehicle.*'
    args.gamma = 2.2
    args.width, args.height = [int(x) for x in args.res.split('x')]
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = WorldSR(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            if not world.tick(clock):
                return
            world.render(display)
            pygame.display.flip()
    finally:
        if world and world.recording_enabled:
            client.stop_recorder()
        if world is not None:
            world.destroy()
        pygame.quit()

def main():
    argparser = argparse.ArgumentParser(description='CARLA No Rendering Mode Visualizer')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--map', metavar='TOWN', default=None, help='start a new episode at the given TOWN')
    argparser.add_argument('--no-rendering', action='store_true', default=True, help='switch off server rendering')
    argparser.add_argument('--show-triggers', action='store_true', help='show trigger boxes of traffic signs')
    argparser.add_argument('--show-connections', action='store_true', help='show waypoint connections')
    argparser.add_argument('--show-spawn-points', action='store_true', help='show recommended spawn points')
    args = argparser.parse_args()
    args.description = argparser.description
    args.width, args.height = [int(x) for x in args.res.split('x')]
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)
    game_loop(args)


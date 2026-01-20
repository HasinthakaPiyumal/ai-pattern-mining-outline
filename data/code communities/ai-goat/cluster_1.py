# Cluster 1

def run(inputs):
    banner()
    if inputs is not None:
        args = handle_args(inputs)
        if args.install is True:
            Installer.install()
            exit()
        if args.run == 'ctfd':
            Runner.ctfd()
            exit()
        if args.run == '1':
            Runner.challenge_1()
            exit()
        if args.run == '2':
            Runner.challenge_2()
            exit()
    handle_args(['--help'])

def banner():
    title = "                                               \n                                                                    \n        _))\n        > *\\     _~\n        `;'\\__-' \\_\n    ____  | )  _ \\ \\ _____________________\n    ____  / / ``  w w ____________________        \n    ____ w w ________________AI_Goat______                                                                          \n    ______________________________________\n\n    Presented by: rootcauz\n\n    "
    print(title)

def handle_args(inputs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--install', help='Install', action='store_true')
    parser.add_argument('-r', '--run', help='Start CTFd or a Challenge.', choices=['ctfd', '1', '2', '3'])
    args = parser.parse_args(inputs)
    return args


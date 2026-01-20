# Cluster 4

def main(args):
    agent_names, task_names, validation_names, details = analyze_output(args.config, args.output, parse_timestamp(args.time))
    task_names.sort(key=lambda x: TaskHandler.get_handler(x).get_order_priority())
    summary = OrderedDict()
    for agent in details:
        summary[agent] = OrderedDict()
        for task in details[agent]:
            handler = TaskHandler.get_handler(task)
            if handler is not None:
                summary[agent][task] = handler.get_main_metric(details[agent][task]['overall'])
            else:
                summary[agent][task] = details[agent][task]['overall']
    for agent in details:
        for task in details[agent]:
            print(ColorMessage.cyan(f'Agent: {agent:20} Task: {task:20} Path: {details[agent][task]['file']}'))
    final_result = {'summary': summary, 'details': details}
    os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(args.save, 'result.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(final_result, f, indent=4, allow_unicode=True, sort_keys=True)
    with open(os.path.join(args.save, 'summary.csv'), 'w', encoding='utf-8') as f:
        '\n        Format:\n            Agent\\Task, Task1, Task2, ...\n            Agent1, MainMetric(Agent1,Task1), MainMetric(Agent1,Task2), ...\n            ......\n        '
        f.write('Agent\\Task,' + ','.join(task_names) + '\n')
        for agent in summary:
            f.write(agent + ',' + ','.join([str(summary[agent][task]) if task in summary[agent] else '' for task in task_names]) + '\n')
    agent_validations = {agent: {validation: [] for validation in validation_names} for agent in agent_names}
    task_validations = {task: {validation: [] for validation in validation_names} for task in task_names}
    for agent in summary:
        for task in summary[agent]:
            if 'validation' in details[agent][task]['overall']:
                for validation in details[agent][task]['overall']['validation']:
                    agent_validations[agent][validation].append(details[agent][task]['overall']['validation'][validation])
                    task_validations[task][validation].append(details[agent][task]['overall']['validation'][validation])
    with open(os.path.join(args.save, 'agent_validation.csv'), 'w', encoding='utf-8') as f:
        '\n        Format:\n            Agent\\Validation, Validation1, Validation2, ...\n            Agent1, Avg(Agent1,Validation1), Avg(Agent1,Validation2), ...\n            ......\n        '
        f.write('Agent\\Validation,' + ','.join(validation_names) + '\n')
        for agent in agent_validations:
            f.write(agent + ',' + ','.join([str(sum(agent_validations[agent][validation]) / len(agent_validations[agent][validation])) if validation in agent_validations[agent] and len(agent_validations[agent][validation]) > 0 else '--' for validation in validation_names]) + '\n')
    with open(os.path.join(args.save, 'task_validation.csv'), 'w', encoding='utf-8') as f:
        '\n        Format:\n            Task\\Validation, Validation1, Validation2, ...\n            Task1, Avg(Task1,Validation1), Avg(Task1,Validation2), ...\n            ......\n        '
        f.write('Task\\Validation,' + ','.join(validation_names) + '\n')
        for task in task_validations:
            f.write(task + ',' + ','.join([str(sum(task_validations[task][validation]) / len(task_validations[task][validation])) if validation in task_validations[task] and len(task_validations[task][validation]) > 0 else '--' for validation in validation_names]) + '\n')
    print(ColorMessage.green(f'Analysis result saved to {os.path.abspath(args.save)}'))

def analyze_output(config: str, output: str, since_timestamp: float):
    """
    Walk through the output folder (including sub-dir) and analyze the overall.json file
    Rule:
        - valid overall file: **/{agent}/{task}/overall.json
        - if a same (agent, task) pair, select the latest one
    """
    loader = ConfigLoader()
    config: dict = loader.load_from(config)
    assert 'definition' in config, 'definition not found in config'
    assert 'agent' in config['definition'], 'agent not found in config.definition'
    assert 'task' in config['definition'], 'task not found in config.definition'
    agents = set(config['definition']['agent'].keys()).intersection(set(MODEL_MAP.keys()))
    tasks = list(config['definition']['task'].keys())
    print(ColorMessage.cyan(f'Available Agents ({len(agents)}):\n    ' + '\n    '.join(agents) + '\n\n' + f'Available Tasks ({len(tasks)}):\n    ' + '\n    '.join(tasks) + '\n'))
    overall_dict = OrderedDict()
    for root, dirs, files in os.walk(output):
        if 'overall.json' in files:
            root = os.path.abspath(root)
            pattern = root.split('/')
            if len(pattern) < 2:
                continue
            agent = pattern[-2]
            task = pattern[-1]
            ct = os.path.getmtime(os.path.join(root, 'overall.json'))
            if agent not in agents:
                continue
            elif task not in tasks:
                continue
            elif ct < since_timestamp:
                continue
            agent = MODEL_MAP[agent]
            if agent in overall_dict and task in overall_dict[agent]:
                if ct < overall_dict[agent][task]['time']:
                    continue
            overall_dict.setdefault(agent, OrderedDict())
            overall_dict[agent][task] = {'file': os.path.join(root, 'overall.json'), 'time': os.path.getmtime(os.path.join(root, 'overall.json'))}
    agent_names = []
    task_names = []
    validation_names = []
    for agent in overall_dict:
        if agent not in agent_names:
            agent_names.append(agent)
        for task in overall_dict[agent]:
            if task not in task_names:
                task_names.append(task)
            overall_dict[agent][task]['time'] = datetime.datetime.fromtimestamp(overall_dict[agent][task]['time']).strftime('%Y-%m-%d %H:%M:%S')
            with open(overall_dict[agent][task]['file'], 'r', encoding='utf-8') as f:
                overall_dict[agent][task]['overall'] = json.load(f)
            if 'validation' in overall_dict[agent][task]['overall']:
                overall_dict[agent][task]['overall']['validation'] = {validation: VALIDATION_MAP_FUNC[validation](overall_dict[agent][task]['overall']['validation']) for validation in VALIDATION_MAP_FUNC}
                for validation in overall_dict[agent][task]['overall']['validation']:
                    if validation not in validation_names:
                        validation_names.append(validation)
    return (agent_names, task_names, validation_names, overall_dict)

def parse_timestamp(time_str: str) -> float:
    try:
        return float(time_str)
    except:
        pass
    try:
        return datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').timestamp()
    except:
        pass
    try:
        return datetime.datetime.strptime(time_str, '%Y-%m-%d').timestamp()
    except:
        pass
    try:
        return datetime.datetime.strptime(time_str, '%Y-%m').timestamp()
    except:
        pass
    num = float(re.findall('[\\d\\.]+', time_str)[0])
    unit = re.findall('[a-zA-Z]+', time_str)[0]
    if unit == 'd':
        delta = num * 24 * 60 * 60
    elif unit == 'h':
        delta = num * 60 * 60
    elif unit == 'm':
        delta = num * 60
    elif unit == 's':
        delta = num
    else:
        raise Exception('Unknown time unit')
    return time.time() - delta


# Cluster 8

def print_spectrum(inpaths, legend, budget=1000000.0, sort=False):
    runs = common.load_runs(inpaths, budget)
    percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
    scores = common.compute_scores(percents)
    if not legend:
        methods = sorted(set((run['method'] for run in runs)))
        legend = {x: x.replace('_', ' ').title() for x in methods}
    scores = np.nanmean(scores, 1)
    percents = np.nanmean(percents, 1)
    if sort:
        first = next(iter(legend.keys()))
        tasks = sorted(tasks, key=lambda task: -np.nanmean(percents[first, task]))
    legend = dict(reversed(legend.items()))
    cols = ''.join((f' & \\textbf{{{k}}}' for k in legend.values()))
    print('\\newcommand{\\o}{\\hphantom{0}}')
    print('\\newcommand{\\b}[1]{\\textbf{#1}}')
    print('')
    print(f'{'Achievement':<20}' + cols + ' \\\\')
    print('')
    wins = collections.defaultdict(int)
    for task in tasks:
        k = tasks.index(task)
        if task.startswith('achievement_'):
            name = task[len('achievement_'):].replace('_', ' ').title()
        else:
            name = task.replace('_', ' ').title()
        print(f'{name:<20}', end='')
        best = max((percents[methods.index(m), k] for m in legend.keys()))
        for method in legend.keys():
            i = methods.index(method)
            value = percents[i][k]
            winner = value >= 0.95 * best and value > 0
            fmt = f'{value:.1f}\\%'
            fmt = ('\\o' if len(fmt) < 6 else ' ') + fmt
            fmt = f'\\b{{{fmt}}}' if winner else f'   {fmt} '
            if winner:
                wins[method] += 1
            print(f' & ${fmt}$', end='')
        print(' \\\\')
    print('')
    print(f'{'Score':<20}', end='')
    best = max((scores[methods.index(m)] for m in legend.keys()))
    for method in legend.keys():
        value = scores[methods.index(method)]
        bold = value >= 0.95 * best and value > 0
        fmt = f'{value:.1f}\\%'
        fmt = ('\\o' if len(fmt) < 6 else ' ') + fmt
        fmt = f'\\b{{{fmt}}}' if bold else f'   {fmt} '
        print(f' & ${fmt}$', end='')
    print(' \\\\')


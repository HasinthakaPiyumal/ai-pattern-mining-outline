# Cluster 5

def plot_spectrum(inpaths, outpath, legend, colors, budget=1000000.0, sort=False):
    runs = common.load_runs(inpaths, budget)
    percents, methods, seeds, tasks = common.compute_success_rates(runs, budget, sortby=sort and legend and list(legend.keys())[0])
    if not legend:
        methods = sorted(set((run['method'] for run in runs)))
        legend = {x: x.replace('_', ' ').title() for x in methods}
    fig, ax = plt.subplots(figsize=(7, 3))
    centers = np.arange(len(tasks))
    width = 0.7
    for index, (method, label) in enumerate(legend.items()):
        heights = np.nanmean(percents[methods.index(method)], 0)
        pos = centers + width * (0.5 / len(methods) + index / len(methods) - 0.5)
        color = colors[index]
        ax.bar(pos, heights, width / len(methods), label=label, color=color)
    names = [x[len('achievement_'):].replace('_', ' ').title() for x in tasks]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', width=14, length=0.8, direction='inout')
    ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
    ax.set_xticks(centers + 0.0)
    ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Success Rate (%)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels('0.01 0.1 1 10 100'.split())
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(loc='upper center', ncol=10, frameon=False, borderpad=0, borderaxespad=0)
    pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath)
    print(f'Saved {outpath}')


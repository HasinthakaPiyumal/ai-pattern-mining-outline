# Cluster 7

def plot_scores(inpaths, outpath, legend, colors, budget=1000000.0, ylim=None):
    runs = common.load_runs(inpaths, budget)
    percents, methods, seeds, tasks = common.compute_success_rates(runs, budget)
    scores = common.compute_scores(percents)
    if not legend:
        methods = sorted(set((run['method'] for run in runs)))
        legend = {x: x.replace('_', ' ').title() for x in methods}
    legend = dict(reversed(legend.items()))
    scores = scores[np.array([methods.index(m) for m in legend.keys()])]
    mean = np.nanmean(scores, -1)
    std = np.nanstd(scores, -1)
    fig, ax = plt.subplots(figsize=(4, 3))
    centers = np.arange(len(legend))
    width = 0.7
    colors = list(reversed(colors[:len(legend)]))
    error_kw = dict(capsize=5, c='#000')
    ax.bar(centers, mean, yerr=std, color=colors, error_kw=error_kw)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', width=50, length=0.8, direction='inout')
    ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
    ax.set_xticks(centers + 0.0)
    ax.set_xticklabels(list(legend.values()), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Crafter Score (%)')
    if ylim:
        ax.set_ylim(0, ylim)
    fig.tight_layout()
    pathlib.Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath)
    print(f'Saved {outpath}')


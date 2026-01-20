# Cluster 47

def compare_1d(real, synth, columns=None, figsize=None):
    """Generate a 1d scatter plot comparing real/synthetic data.

    Args:
        real (pd.DataFrame):
            The real data.
        synth (pd.DataFrame):
            The synthetic data.
        columns (list):
            The name of the columns to plot.
        figsize:
            Figure size, passed to matplotlib.
    """
    if len(real.shape) == 1:
        real = pd.DataFrame({'': real})
        synth = pd.DataFrame({'': synth})
    columns = columns or real.columns
    num_cols = len(columns)
    fig_cols = min(2, num_cols)
    fig_rows = num_cols // fig_cols + 1
    prefix = f'{fig_rows}{fig_cols}'
    figsize = figsize or (5 * fig_cols, 3 * fig_rows)
    fig = plt.figure(figsize=figsize)
    for idx, column in enumerate(columns):
        position = int(prefix + str(idx + 1))
        hist_1d(real[column], fig=fig, position=position, title=column, label='Real')
        hist_1d(synth[column], fig=fig, position=position, title=column, label='Synthetic')
    plt.tight_layout()

def hist_1d(data, fig=None, title=None, position=None, bins=20, label=None):
    """Plot 1 dimensional data in a histogram."""
    fig = fig or plt.figure()
    position = position or 111
    ax = fig.add_subplot(position)
    ax.hist(data, density=True, bins=bins, alpha=0.8, label=label)
    if label:
        ax.legend()
    if title:
        ax.set_title(title)
        ax.title.set_position([0.5, 1.05])
    return ax


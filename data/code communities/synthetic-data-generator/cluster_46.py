# Cluster 46

def compare_3d(real, synth, columns=None, figsize=(10, 4)):
    """Generate a 3d scatter plot comparing real/synthetic data.

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
    columns = columns or real.columns
    fig = plt.figure(figsize=figsize)
    scatter_3d(real[columns], fig=fig, title='Real Data', position=121)
    scatter_3d(synth[columns], fig=fig, title='Synthetic Data', position=122)
    plt.tight_layout()

def scatter_3d(data, columns=None, fig=None, title=None, position=None):
    """Plot 3 dimensional data in a scatter plot."""
    fig = fig or plt.figure()
    position = position or 111
    ax = fig.add_subplot(position, projection='3d')
    ax.scatter(*(data[column] for column in columns or data.columns))
    if title:
        ax.set_title(title)
        ax.title.set_position([0.5, 1.05])
    return ax


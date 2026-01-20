# Cluster 2

def visualize_latent_space(latent_dim, n_samples, decoder):
    latents = ptu.FloatTensor(sample_random_normal(latent_dim, n_samples))
    pred_rewards = ptu.get_numpy(decoder(latents, None))
    goal_locations = np.argmax(pred_rewards, axis=-1)
    if latent_dim > 2:
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(latents)
    data = tsne_results if latent_dim > 2 else latents
    df = pd.DataFrame(data, columns=['x1', 'x2'])
    df['y'] = goal_locations
    fig = plt.figure(figsize=(6, 6))
    sns.scatterplot(x='x1', y='x2', hue='y', s=30, palette=sns.color_palette('hls', len(np.unique(df['y']))), data=df, legend='full', ax=plt.gca())
    fig.show()
    return (data, goal_locations)

def sample_random_normal(dim, n_samples):
    return np.random.normal(size=(n_samples, dim))


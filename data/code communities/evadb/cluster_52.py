# Cluster 52

def plot(df):
    import matplotlib.pyplot as plt
    grouped = df.groupby(df.columns[3]).size().reset_index(name='count')
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped.plot(kind='bar', x=grouped.columns[0], y=grouped.columns[1], ax=ax, legend=False, color='#ed7d32', width=0.6)
    plt.title('Concentration of recreational parks', fontsize=18)
    plt.xlabel('SF neighbourhood', fontsize=14)
    plt.ylabel('Park count', fontsize=14)
    plt.xticks(rotation=75, fontsize=8)
    plt.tight_layout()
    plt.show()


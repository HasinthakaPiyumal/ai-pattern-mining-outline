# Cluster 4

def read_stats(indir, outdir, task, method, budget=int(1000000.0), verbose=False):
    indir = pathlib.Path(indir)
    outdir = pathlib.Path(outdir)
    runs = []
    print(f'Loading {indir.name}...')
    filenames = sorted(list(indir.glob('**/stats.jsonl')))
    for index, filename in enumerate(filenames):
        if not filename.is_file():
            continue
        rewards, lengths, achievements = load_stats(filename, budget)
        if sum(lengths) < budget - 10000.0:
            message = f'Skipping incomplete run ({sum(lengths)} < {budget} steps): '
            message += f'{filename.relative_to(indir.parent)}'
            print(f'==> {message}')
            continue
        runs.append(dict(task=task, method=method, seed=str(index), xs=np.cumsum(lengths).tolist(), reward=rewards, length=lengths, **achievements))
    if not runs:
        print('No completed runs.\n')
        return
    print_summary(runs, budget, verbose)
    outdir.mkdir(exist_ok=True, parents=True)
    filename = outdir / f'{task}-{method}.json'
    filename.write_text(json.dumps(runs))
    print('Wrote', filename)
    print('')


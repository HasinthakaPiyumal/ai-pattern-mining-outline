# Cluster 11

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {'black': '\x1b[30m', 'red': '\x1b[31m', 'green': '\x1b[32m', 'yellow': '\x1b[33m', 'blue': '\x1b[34m', 'magenta': '\x1b[35m', 'cyan': '\x1b[36m', 'white': '\x1b[37m', 'bright_black': '\x1b[90m', 'bright_red': '\x1b[91m', 'bright_green': '\x1b[92m', 'bright_yellow': '\x1b[93m', 'bright_blue': '\x1b[94m', 'bright_magenta': '\x1b[95m', 'bright_cyan': '\x1b[96m', 'bright_white': '\x1b[97m', 'end': '\x1b[0m', 'bold': '\x1b[1m', 'underline': '\x1b[4m'}
    return ''.join((colors[x] for x in args)) + f'{string}' + colors['end']

def plot_evolve(evolve_csv='path/to/evolve.csv'):
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=0.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})
        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')
    f = evolve_csv.with_suffix('.png')
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')

def fitness(x):
    w = [0.0, 0.0, 0.1, 0.9]
    return (x[:, :4] * w).sum(1)

def hist2d(x, y, n=100):
    xedges, yedges = (np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n))
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

def print_mutation(results, hyp, save_dir, bucket):
    evolve_csv, results_csv, evolve_yaml = (save_dir / 'evolve.csv', save_dir / 'results.csv', save_dir / 'hyp_evolve.yaml')
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())
    keys = tuple((x.strip() for x in keys))
    vals = results + tuple(hyp.values())
    n = len(keys)
    if bucket:
        url = f'gs://{bucket}/evolve.csv'
        if gsutil_getsize(url) > (os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0):
            os.system(f'gsutil cp {url} {save_dir}')
    s = '' if evolve_csv.exists() else ('%20s,' * n % keys).rstrip(',') + '\n'
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')
    print(colorstr('evolve: ') + ', '.join((f'{x.strip():>20s}' for x in keys)))
    print(colorstr('evolve: ') + ', '.join((f'{x:20.5g}' for x in vals)), end='\n\n\n')
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())
        i = np.argmax(fitness(data.values[:, :7]))
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' + f'# Best generation: {i}\n' + f'# Last generation: {len(data) - 1}\n' + '# ' + ', '.join((f'{x.strip():>20s}' for x in keys[:7])) + '\n' + '# ' + ', '.join((f'{x:>20.5g}' for x in data.values[i, :7])) + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)
    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')

def gsutil_getsize(url=''):
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0

class Loggers:

    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger
        self.include = include
        self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'x/lr0', 'x/lr1', 'x/lr2']
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
        for k in LOGGERS:
            setattr(self, k, None)
        self.csv = True
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            print(emojis(s))
        s = self.save_dir
        if 'tb' in self.include and (not self.opt.evolve):
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and (not wandb_artifact_resume) else None
            self.opt.hyp = self.hyp
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

    def on_pretrain_routine_end(self):
        paths = self.save_dir.glob('*labels*.jpg')
        if self.wandb:
            self.wandb.log({'Labels': [wandb.Image(str(x), caption=x.name) for x in paths]})

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, sync_bn):
        if plots:
            if ni == 0:
                if not sync_bn:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
            if self.wandb and ni == 10:
                files = sorted(self.save_dir.glob('train*.jpg'))
                self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})

    def on_train_epoch_end(self, epoch):
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_val_image_end(self, pred, predn, path, names, im):
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)

    def on_val_end(self):
        if self.wandb:
            files = sorted(self.save_dir.glob('val*.jpg'))
            self.wandb.log({'Validation': [wandb.Image(str(f), caption=f.name) for f in files]})

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        x = {k: v for k, v in zip(self.keys, vals)}
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1
            s = '' if file.exists() else ('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n'
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')
        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)
        if self.wandb:
            if best_fitness == fi:
                best_results = [epoch] + vals[3:7]
                for i, name in enumerate(self.best_keys):
                    self.wandb.wandb_run.summary[name] = best_results[i]
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and (not final_epoch)) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, plots, epoch, results):
        if plots:
            plot_results(file=self.save_dir / 'results.csv')
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [self.save_dir / f for f in files if (self.save_dir / f).exists()]
        if self.tb:
            import cv2
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')
        if self.wandb:
            self.wandb.log({k: v for k, v in zip(self.keys[3:10], results)})
            self.wandb.log({'Results': [wandb.Image(str(f), caption=f.name) for f in files]})
            if not self.opt.evolve:
                wandb.log_artifact(str(best if best.exists() else last), type='model', name='run_' + self.wandb.wandb_run.id + '_model', aliases=['latest', 'best', 'stripped'])
                self.wandb.finish_run()
            else:
                self.wandb.finish_run()
                self.wandb = WandbLogger(self.opt)

    def on_params_update(self, params):
        if self.wandb:
            self.wandb.wandb_run.config.update(params, allow_val_change=True)

def plot_results(file='path/to/results.csv', dir=''):
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j]
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()


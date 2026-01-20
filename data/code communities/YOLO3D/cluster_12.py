# Cluster 12

def check_anchors(dataset, model, thr=4.0, imgsz=640):
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1 / thr).float().sum(1).mean()
        bpr = (best > 1 / thr).float().mean()
        return (bpr, aat)
    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:
        LOGGER.info(emojis(f'{s}Current anchors are a good fit to dataset ✅'))
    else:
        LOGGER.info(emojis(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...'))
        na = m.anchors.numel() // 2
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f'{PREFIX}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)
            check_anchor_order(m)
            LOGGER.info(f'{PREFIX}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            LOGGER.info(f'{PREFIX}Original anchors better than new anchors. Proceeding with original anchors.')

def emojis(str=''):
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)
        assert file.exists(), f'{prefix} {file.resolve()} not found, check failed.'
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    else:
        requirements = [x for x in requirements if x not in exclude]
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:
            s = f'{prefix} {r} not found and is required by YOLOv5'
            if install:
                print(f'{s}, attempting auto-update...')
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')
    if n:
        source = file.resolve() if 'file' in locals() else requirements
        s = f'{prefix} {n} package{'s' * (n > 1)} updated per {source}\n{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n'
        print(emojis(s))

def notebook_init(verbose=True):
    print('Checking setup...')
    import os
    import shutil
    from utils.general import check_requirements, emojis, is_colab
    from utils.torch_utils import select_device
    check_requirements(('psutil', 'IPython'))
    import psutil
    from IPython import display
    if is_colab():
        shutil.rmtree('/content/sample_data', ignore_errors=True)
    if verbose:
        gib = 1 / 1024 ** 3
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        display.clear_output()
        s = f'({os.cpu_count()} CPUs, {ram * gib:.1f} GB RAM, {(total - free) * gib:.1f}/{total * gib:.1f} GB disk)'
    else:
        s = ''
    select_device(newline=False)
    print(emojis(f'Setup complete ✅ {s}'))
    return display

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def set_logging(name=None, verbose=True):
    for h in logging.root.handlers:
        logging.root.removeHandler(h)
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format='%(message)s', level=logging.INFO if verbose and rank in (-1, 0) else logging.WARNING)
    return logging.getLogger(name)

@try_except
@WorkingDirectory(ROOT)
def check_git_status():
    msg = ', for updates see https://github.com/ultralytics/yolov5'
    print(colorstr('github: '), end='')
    assert Path('.git').exists(), 'skipping check (not a git repository)' + msg
    assert not is_docker(), 'skipping check (Docker image)' + msg
    assert check_online(), 'skipping check (offline)' + msg
    cmd = 'git fetch && git config --get remote.origin.url'
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))
    if n > 0:
        s = f'⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update.'
    else:
        s = f'up to date with {url} ✅'
    print(emojis(s))

def is_docker():
    return Path('/workspace').exists()

def check_online():
    import socket
    try:
        socket.create_connection(('1.1.1.1', 443), 5)
        return True
    except OSError:
        return False

def check_python(minimum='3.6.2'):
    check_version(platform.python_version(), minimum, name='Python ', hard=True)

def check_imshow():
    try:
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


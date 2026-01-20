# Cluster 15

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model of FPS')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path')
    parser.add_argument('--round-num', type=int, default=1, help='round a number to a given precision in decimal digits')
    parser.add_argument('--repeat-num', type=int, default=1, help='number of repeat times of measurement for averaging the results')
    parser.add_argument('--out', type=str, help='output path of gathered fps to be stored')
    parser.add_argument('--max-iter', type=int, default=2000, help='num of max iter')
    parser.add_argument('--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='Whether to fuse conv and bn, this will slightly increasethe inference speed')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def results2markdown(result_dict):
    table_data = []
    is_multiple_results = False
    for cfg_name, value in result_dict.items():
        name = cfg_name.replace('configs/', '')
        fps = value['fps']
        ms_times_pre_image = value['ms_times_pre_image']
        if isinstance(fps, list):
            is_multiple_results = True
            mean_fps = value['mean_fps']
            mean_times_pre_image = value['mean_times_pre_image']
            fps_str = ','.join([str(s) for s in fps])
            ms_times_pre_image_str = ','.join([str(s) for s in ms_times_pre_image])
            table_data.append([name, fps_str, mean_fps, ms_times_pre_image_str, mean_times_pre_image])
        else:
            table_data.append([name, fps, ms_times_pre_image])
    if is_multiple_results:
        table_data.insert(0, ['model', 'fps', 'mean_fps', 'times_pre_image(ms)', 'mean_times_pre_image(ms)'])
    else:
        table_data.insert(0, ['model', 'fps', 'times_pre_image(ms)'])
    table = GithubFlavoredMarkdownTable(table_data)
    print(table.table, flush=True)

def main():
    args = parse_args()
    logger = get_logger(name='mmdet', log_file=args.out)
    if args.https_proxy:
        os.environ['https_proxy'] = args.https_proxy
    http_session = requests.Session()
    for resource_prefix in ('http://', 'https://'):
        http_session.mount(resource_prefix, requests.adapters.HTTPAdapter(max_retries=5, pool_connections=20, pool_maxsize=args.num_threads))
    logger.info('Finding all markdown files in the current directory...')
    project_root = (pathlib.Path(__file__).parent / '..').resolve()
    markdown_files = project_root.glob('**/*.md')
    all_matches = set()
    url_regex = re.compile('\\[([^!][^\\]]+)\\]\\(([^)(]+)\\)')
    for markdown_file in markdown_files:
        with open(markdown_file) as handle:
            for line in handle.readlines():
                matches = url_regex.findall(line)
                for name, link in matches:
                    if 'localhost' not in link:
                        all_matches.add(MatchTuple(source=str(markdown_file), name=name, link=link))
    logger.info(f'  {len(all_matches)} markdown files found')
    logger.info('Checking to make sure we can retrieve each link...')
    with Pool(processes=args.num_threads) as pool:
        results = pool.starmap(check_link, [(match, http_session, logger) for match in list(all_matches)])
    unreachable_results = [(match_tuple, reason) for match_tuple, success, reason in results if not success]
    if unreachable_results:
        logger.info('================================================')
        logger.info(f'Unreachable links ({len(unreachable_results)}):')
        for match_tuple, reason in unreachable_results:
            logger.info('  > Source: ' + match_tuple.source)
            logger.info('    Name: ' + match_tuple.name)
            logger.info('    Link: ' + match_tuple.link)
            if reason is not None:
                logger.info('    Reason: ' + reason)
        sys.exit(1)
    logger.info('No Unreachable link found.')

def main():
    args = parse_args()
    benchmark_type = []
    if args.basic_arch:
        benchmark_type += basic_arch_root
    if args.datasets:
        benchmark_type += datasets_root
    if args.data_pipeline:
        benchmark_type += data_pipeline_root
    if args.nn_module:
        benchmark_type += nn_module_root
    special_model = args.model_options
    if special_model is not None:
        benchmark_type += special_model
    config_dpath = 'configs/'
    benchmark_configs = []
    for cfg_root in benchmark_type:
        cfg_dir = osp.join(config_dpath, cfg_root)
        configs = os.scandir(cfg_dir)
        for cfg in configs:
            config_path = osp.join(cfg_dir, cfg.name)
            if config_path in benchmark_pool and config_path not in benchmark_configs:
                benchmark_configs.append(config_path)
    print(f'Totally found {len(benchmark_configs)} configs to benchmark')
    with open(args.out, 'w') as f:
        for config in benchmark_configs:
            f.write(config + '\n')

def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), f'Expected out file path suffix is .sh, but get .{out_suffix}'
    assert args.out or args.run, 'Please specify at least one operation (save/run/ the script) with the argument "--out" or "--run"'
    partition = args.partition
    root_name = './tools'
    train_script_name = osp.join(root_name, 'slurm_train.sh')
    stdout_cfg = '>/dev/null'
    max_keep_ckpts = args.max_keep_ckpts
    commands = []
    with open(args.txt_path, 'r') as f:
        model_cfgs = f.readlines()
        for i, cfg in enumerate(model_cfgs):
            cfg = cfg.strip()
            if len(cfg) == 0:
                continue
            echo_info = f"echo '{cfg}' &"
            commands.append(echo_info)
            commands.append('\n')
            fname, _ = osp.splitext(osp.basename(cfg))
            out_fname = osp.join(root_name, 'work_dir', fname)
            if cfg.find('16x') >= 0:
                command_info = f'GPUS=16  GPUS_PER_NODE=8  CPUS_PER_TASK=2 {train_script_name} '
            elif cfg.find('gn-head_4x4_1x_coco.py') >= 0 or cfg.find('gn-head_4x4_2x_coco.py') >= 0:
                command_info = f'GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 {train_script_name} '
            else:
                command_info = f'GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 {train_script_name} '
            command_info += f'{partition} '
            command_info += f'{fname} '
            command_info += f'{cfg} '
            command_info += f'{out_fname} '
            if max_keep_ckpts:
                command_info += f'--cfg-options checkpoint_config.max_keep_ckpts={max_keep_ckpts}' + ' '
            command_info += f'{stdout_cfg} &'
            commands.append(command_info)
            if i < len(model_cfgs):
                commands.append('\n')
        command_str = ''.join(commands)
        if args.out:
            with open(args.out, 'w') as f:
                f.write(command_str)
        if args.run:
            os.system(command_str)

def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), f'Expected out file path suffix is .sh, but get .{out_suffix}'
    assert args.out or args.run, 'Please specify at least one operation (save/run/ the script) with the argument "--out" or "--run"'
    commands = []
    partition_name = 'PARTITION=$1 '
    commands.append(partition_name)
    commands.append('\n')
    checkpoint_root = 'CHECKPOINT_DIR=$2 '
    commands.append(checkpoint_root)
    commands.append('\n')
    script_name = osp.join('tools', 'slurm_test.sh')
    port = args.port
    work_dir = args.work_dir
    cfg = Config.fromfile(args.config)
    for model_key in cfg:
        model_infos = cfg[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print('processing: ', model_info['config'])
            model_test_dict = process_model_info(model_info, work_dir)
            create_test_bash_info(commands, model_test_dict, port, script_name, '$PARTITION')
            port += 1
    command_str = ''.join(commands)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(command_str)
    if args.run:
        os.system(command_str)

def process_model_info(model_info, work_dir):
    config = model_info['config'].strip()
    fname, _ = osp.splitext(osp.basename(config))
    job_name = fname
    work_dir = osp.join(work_dir, fname)
    checkpoint = model_info['checkpoint'].strip()
    if not isinstance(model_info['eval'], list):
        evals = [model_info['eval']]
    else:
        evals = model_info['eval']
    eval = ' '.join(evals)
    return dict(config=config, job_name=job_name, work_dir=work_dir, checkpoint=checkpoint, eval=eval)

def create_test_bash_info(commands, model_test_dict, port, script_name, partition):
    config = model_test_dict['config']
    job_name = model_test_dict['job_name']
    checkpoint = model_test_dict['checkpoint']
    work_dir = model_test_dict['work_dir']
    eval = model_test_dict['eval']
    echo_info = f" \necho '{config}' &"
    commands.append(echo_info)
    commands.append('\n')
    command_info = f'GPUS=8  GPUS_PER_NODE=8  CPUS_PER_TASK=2 {script_name} '
    command_info += f'{partition} '
    command_info += f'{job_name} '
    command_info += f'{config} '
    command_info += f'$CHECKPOINT_DIR/{checkpoint} '
    command_info += f'--work-dir {work_dir} '
    command_info += f'--eval {eval} '
    command_info += f'--cfg-option dist_params.port={port} '
    command_info += ' &'
    commands.append(command_info)

def get_final_results(log_json_path, epoch_or_iter, results_lut, by_epoch=True):
    result_dict = dict()
    last_val_line = None
    last_train_line = None
    last_val_line_idx = -1
    last_train_line_idx = -1
    with open(log_json_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            log_line = json.loads(line)
            if 'mode' not in log_line.keys():
                continue
            if by_epoch:
                if log_line['mode'] == 'train' and log_line['epoch'] == epoch_or_iter:
                    result_dict['memory'] = log_line['memory']
                if log_line['mode'] == 'val' and log_line['epoch'] == epoch_or_iter:
                    result_dict.update({key: log_line[key] for key in results_lut if key in log_line})
                    return result_dict
            else:
                if log_line['mode'] == 'train':
                    last_train_line_idx = i
                    last_train_line = log_line
                if log_line and log_line['mode'] == 'val':
                    last_val_line_idx = i
                    last_val_line = log_line
    assert last_val_line_idx == last_train_line_idx + 1, 'Log file is incomplete'
    result_dict['memory'] = last_train_line['memory']
    result_dict.update({key: last_val_line[key] for key in results_lut if key in last_val_line})
    return result_dict

def convert_model_info_to_pwc(model_infos):
    pwc_files = {}
    for model in model_infos:
        cfg_folder_name = osp.split(model['config'])[-2]
        pwc_model_info = OrderedDict()
        pwc_model_info['Name'] = osp.split(model['config'])[-1].split('.')[0]
        pwc_model_info['In Collection'] = 'Please fill in Collection name'
        pwc_model_info['Config'] = osp.join('configs', model['config'])
        memory = round(model['results']['memory'] / 1024, 1)
        meta_data = OrderedDict()
        meta_data['Training Memory (GB)'] = memory
        if 'epochs' in model:
            meta_data['Epochs'] = get_real_epoch_or_iter(model['config'])
        else:
            meta_data['Iterations'] = get_real_epoch_or_iter(model['config'])
        pwc_model_info['Metadata'] = meta_data
        dataset_name = get_dataset_name(model['config'])
        results = []
        if 'bbox_mAP' in model['results']:
            metric = round(model['results']['bbox_mAP'] * 100, 1)
            results.append(OrderedDict(Task='Object Detection', Dataset=dataset_name, Metrics={'box AP': metric}))
        if 'segm_mAP' in model['results']:
            metric = round(model['results']['segm_mAP'] * 100, 1)
            results.append(OrderedDict(Task='Instance Segmentation', Dataset=dataset_name, Metrics={'mask AP': metric}))
        if 'PQ' in model['results']:
            metric = round(model['results']['PQ'], 1)
            results.append(OrderedDict(Task='Panoptic Segmentation', Dataset=dataset_name, Metrics={'PQ': metric}))
        pwc_model_info['Results'] = results
        link_string = 'https://download.openmmlab.com/mmdetection/v2.0/'
        link_string += '{}/{}'.format(model['config'].rstrip('.py'), osp.split(model['model_path'])[-1])
        pwc_model_info['Weights'] = link_string
        if cfg_folder_name in pwc_files:
            pwc_files[cfg_folder_name].append(pwc_model_info)
        else:
            pwc_files[cfg_folder_name] = [pwc_model_info]
    return pwc_files

def get_real_epoch_or_iter(config):
    cfg = mmcv.Config.fromfile('./configs/' + config)
    if cfg.runner.type == 'EpochBasedRunner':
        epoch = cfg.runner.max_epochs
        if cfg.data.train.type == 'RepeatDataset':
            epoch *= cfg.data.train.times
        return epoch
    else:
        return cfg.runner.max_iters

def get_dataset_name(config):
    name_map = dict(CityscapesDataset='Cityscapes', CocoDataset='COCO', CocoPanopticDataset='COCO', DeepFashionDataset='Deep Fashion', LVISV05Dataset='LVIS v0.5', LVISV1Dataset='LVIS v1', VOCDataset='Pascal VOC', WIDERFaceDataset='WIDER Face', OpenImagesDataset='OpenImagesDataset', OpenImagesChallengeDataset='OpenImagesChallengeDataset')
    cfg = mmcv.Config.fromfile('./configs/' + config)
    return name_map[cfg.dataset_type]

def main():
    args = parse_args()
    models_root = args.root
    models_out = args.out
    mmcv.mkdir_or_exist(models_out)
    raw_configs = list(mmcv.scandir('./configs', '.py', recursive=True))
    used_configs = []
    for raw_config in raw_configs:
        if osp.exists(osp.join(models_root, raw_config)):
            used_configs.append(raw_config)
    print(f'Find {len(used_configs)} models to be gathered')
    model_infos = []
    for used_config in used_configs:
        exp_dir = osp.join(models_root, used_config)
        by_epoch = is_by_epoch(used_config)
        if args.best is True:
            final_model, final_epoch_or_iter = get_best_epoch_or_iter(exp_dir)
        else:
            final_epoch_or_iter = get_final_epoch_or_iter(used_config)
            final_model = '{}_{}.pth'.format('epoch' if by_epoch else 'iter', final_epoch_or_iter)
        model_path = osp.join(exp_dir, final_model)
        if not osp.exists(model_path):
            continue
        log_json_path = list(sorted(glob.glob(osp.join(exp_dir, '*.log.json'))))[-1]
        log_txt_path = list(sorted(glob.glob(osp.join(exp_dir, '*.log'))))[-1]
        cfg = mmcv.Config.fromfile('./configs/' + used_config)
        results_lut = cfg.evaluation.metric
        if not isinstance(results_lut, list):
            results_lut = [results_lut]
        for i, key in enumerate(results_lut):
            if 'mAP' not in key and 'PQ' not in key:
                results_lut[i] = key + '_mAP'
        model_performance = get_final_results(log_json_path, final_epoch_or_iter, results_lut, by_epoch)
        if model_performance is None:
            continue
        model_time = osp.split(log_txt_path)[-1].split('.')[0]
        model_info = dict(config=used_config, results=model_performance, model_time=model_time, final_model=final_model, log_json_path=osp.split(log_json_path)[-1])
        model_info['epochs' if by_epoch else 'iterations'] = final_epoch_or_iter
        model_infos.append(model_info)
    publish_model_infos = []
    for model in model_infos:
        model_publish_dir = osp.join(models_out, model['config'].rstrip('.py'))
        mmcv.mkdir_or_exist(model_publish_dir)
        model_name = osp.split(model['config'])[-1].split('.')[0]
        model_name += '_' + model['model_time']
        publish_model_path = osp.join(model_publish_dir, model_name)
        trained_model_path = osp.join(models_root, model['config'], model['final_model'])
        final_model_path = process_checkpoint(trained_model_path, publish_model_path)
        shutil.copy(osp.join(models_root, model['config'], model['log_json_path']), osp.join(model_publish_dir, f'{model_name}.log.json'))
        shutil.copy(osp.join(models_root, model['config'], model['log_json_path'].rstrip('.json')), osp.join(model_publish_dir, f'{model_name}.log'))
        config_path = model['config']
        config_path = osp.join('configs', config_path) if 'configs' not in config_path else config_path
        target_config_path = osp.split(config_path)[-1]
        shutil.copy(config_path, osp.join(model_publish_dir, target_config_path))
        model['model_path'] = final_model_path
        publish_model_infos.append(model)
    models = dict(models=publish_model_infos)
    print(f'Totally gathered {len(publish_model_infos)} models')
    mmcv.dump(models, osp.join(models_out, 'model_info.json'))
    pwc_files = convert_model_info_to_pwc(publish_model_infos)
    for name in pwc_files:
        with open(osp.join(models_out, name + '_metafile.yml'), 'w') as f:
            ordered_yaml_dump(pwc_files[name], f, encoding='utf-8')

def is_by_epoch(config):
    cfg = mmcv.Config.fromfile('./configs/' + config)
    return cfg.runner.type == 'EpochBasedRunner'

def get_best_epoch_or_iter(exp_dir):
    best_epoch_iter_full_path = list(sorted(glob.glob(osp.join(exp_dir, 'best_*.pth'))))[-1]
    best_epoch_or_iter_model_path = best_epoch_iter_full_path.split('/')[-1]
    best_epoch_or_iter = best_epoch_or_iter_model_path.split('_')[-1].split('.')[0]
    return (best_epoch_or_iter_model_path, int(best_epoch_or_iter))

def get_final_epoch_or_iter(config):
    cfg = mmcv.Config.fromfile('./configs/' + config)
    if cfg.runner.type == 'EpochBasedRunner':
        return cfg.runner.max_epochs
    else:
        return cfg.runner.max_iters

def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    for key in list(checkpoint['state_dict']):
        if key.startswith('ema_'):
            checkpoint['state_dict'].pop(key)
    if torch.__version__ >= '1.6':
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])
    return final_file

def ordered_yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):

    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

def mmdet2torchserve(config_file: str, checkpoint_file: str, output_folder: str, model_name: str, model_version: str='1.0', force: bool=False):
    """Converts MMDetection model (config + checkpoint) to TorchServe `.mar`.

    Args:
        config_file:
            In MMDetection config format.
            The contents vary for each task repository.
        checkpoint_file:
            In MMDetection checkpoint format.
            The contents vary for each task repository.
        output_folder:
            Folder where `{model_name}.mar` will be created.
            The file created will be in TorchServe archive format.
        model_name:
            If not None, used for naming the `{model_name}.mar` file
            that will be created under `output_folder`.
            If None, `{Path(checkpoint_file).stem}` will be used.
        model_version:
            Model's version.
        force:
            If True, if there is an existing `{model_name}.mar`
            file under `output_folder` it will be overwritten.
    """
    mmcv.mkdir_or_exist(output_folder)
    config = mmcv.Config.fromfile(config_file)
    with TemporaryDirectory() as tmpdir:
        config.dump(f'{tmpdir}/config.py')
        args = Namespace(**{'model_file': f'{tmpdir}/config.py', 'serialized_file': checkpoint_file, 'handler': f'{Path(__file__).parent}/mmdet_handler.py', 'model_name': model_name or Path(checkpoint_file).stem, 'version': model_version, 'export_path': output_folder, 'force': force, 'requirements_file': None, 'extra_files': None, 'runtime': 'python', 'archive_format': 'default'})
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)

def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)
    img_dir = osp.join(cityscapes_path, args.img_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir)
    set_name = dict(train='instancesonly_filtered_gtFine_train.json', val='instancesonly_filtered_gtFine_val.json', test='instancesonly_filtered_gtFine_test.json')
    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(print_tmpl='It took {}s to convert Cityscapes annotation'):
            files = collect_files(osp.join(img_dir, split), osp.join(gt_dir, split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))

def collect_files(img_dir, gt_dir):
    suffix = 'leftImg8bit.png'
    files = []
    for img_file in glob.glob(osp.join(img_dir, '**/*.png')):
        assert img_file.endswith(suffix), img_file
        inst_file = gt_dir + img_file[len(img_dir):-len(suffix)] + 'gtFine_instanceIds.png'
        segm_file = gt_dir + img_file[len(img_dir):-len(suffix)] + 'gtFine_labelIds.png'
        files.append((img_file, inst_file, segm_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')
    return files

def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)
    return images

def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    for label in CSLabels.labels:
        if label.hasInstances and (not label.ignoreInEval):
            cat = dict(id=label.id, name=label.name)
            out_json['categories'].append(cat)
    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')
    mmcv.dump(out_json, out_json_name)
    return out_json

def main():
    args = parse_args()
    assert args.out.endswith('json'), 'The output file name must be json suffix'
    img_infos = collect_image_infos(args.img_path, args.exclude_extensions)
    classes = mmcv.list_from_file(args.classes)
    coco_info = cvt_to_coco_json(img_infos, classes)
    save_dir = os.path.join(args.img_path, '..', 'annotations')
    mmcv.mkdir_or_exist(save_dir)
    save_path = os.path.join(save_dir, args.out)
    mmcv.dump(coco_info, save_path)
    print(f'save json file: {save_path}')

def collect_image_infos(path, exclude_extensions=None):
    img_infos = []
    images_generator = mmcv.scandir(path, recursive=True)
    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (exclude_extensions is not None and (not image_path.lower().endswith(exclude_extensions))):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {'filename': image_path, 'width': img_pillow.width, 'height': img_pillow.height}
            img_infos.append(img_info)
    return img_infos

def cvt_to_coco_json(img_infos, classes):
    image_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()
    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)
    for img_dict in img_infos:
        file_name = img_dict['filename']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)
        image_id += 1
    return coco

def cvt_annotations(devkit_path, years, split, out_file):
    if not isinstance(years, list):
        years = [years]
    annotations = []
    for year in years:
        filelist = osp.join(devkit_path, f'VOC{year}/ImageSets/Main/{split}.txt')
        if not osp.isfile(filelist):
            print(f'filelist does not exist: {filelist}, skip voc{year} {split}')
            return
        img_names = mmcv.list_from_file(filelist)
        xml_paths = [osp.join(devkit_path, f'VOC{year}/Annotations/{img_name}.xml') for img_name in img_names]
        img_paths = [f'VOC{year}/JPEGImages/{img_name}.jpg' for img_name in img_names]
        part_annotations = mmcv.track_progress(parse_xml, list(zip(xml_paths, img_paths)))
        annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    mmcv.dump(annotations, out_file)
    return annotations

def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mmcv.mkdir_or_exist(out_dir)
    years = []
    if osp.isdir(osp.join(devkit_path, 'VOC2007')):
        years.append('2007')
    if osp.isdir(osp.join(devkit_path, 'VOC2012')):
        years.append('2012')
    if '2007' in years and '2012' in years:
        years.append(['2007', '2012'])
    if not years:
        raise IOError(f'The devkit path {devkit_path} contains neither "VOC2007" nor "VOC2012" subfolder')
    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'
    for year in years:
        if year == '2007':
            prefix = 'voc07'
        elif year == '2012':
            prefix = 'voc12'
        elif year == ['2007', '2012']:
            prefix = 'voc0712'
        for split in ['train', 'val', 'trainval']:
            dataset_name = prefix + '_' + split
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, split, osp.join(out_dir, dataset_name + out_fmt))
        if not isinstance(year, list):
            dataset_name = prefix + '_test'
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, 'test', osp.join(out_dir, dataset_name + out_fmt))
    print('Done!')

def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)

def main():
    args = parse_args()
    data_root = args.data_root
    val_info = mmcv.load(osp.join(data_root, 'panoptic_val2017.json'))
    test_old_info = mmcv.load(osp.join(data_root, 'image_info_test-dev2017.json'))
    test_info = test_old_info
    test_info.update({'categories': val_info['categories']})
    mmcv.dump(test_info, osp.join(data_root, 'panoptic_image_info_test-dev2017.json'))

def download(url, dir, unzip=True, delete=False, threads=1):

    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print('Downloading {} to {}'.format(url, f))
            torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.tar'):
            print('Unzipping {}'.format(f.name))
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.tar':
                TarFile(f).extractall(path=dir)
            if delete:
                f.unlink()
                print('Delete {}'.format(f))
    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

def download_one(url, dir):
    f = dir / Path(url).name
    if Path(url).is_file():
        Path(url).rename(f)
    elif not f.exists():
        print('Downloading {} to {}'.format(url, f))
        torch.hub.download_url_to_file(url, f, progress=True)
    if unzip and f.suffix in ('.zip', '.tar'):
        print('Unzipping {}'.format(f.name))
        if f.suffix == '.zip':
            ZipFile(f).extractall(path=dir)
        elif f.suffix == '.tar':
            TarFile(f).extractall(path=dir)
        if delete:
            f.unlink()
            print('Delete {}'.format(f))

def main():
    args = parse_args()
    path = Path(args.save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data2url = dict(coco2017=['http://images.cocodataset.org/zips/train2017.zip', 'http://images.cocodataset.org/zips/val2017.zip', 'http://images.cocodataset.org/zips/test2017.zip', 'http://images.cocodataset.org/annotations/' + 'annotations_trainval2017.zip'], lvis=['https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip', 'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip'], voc2007=['http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar', 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar', 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar'])
    url = data2url.get(args.dataset_name, None)
    if url is None:
        print('Only support COCO, VOC, and LVIS now!')
        return
    download(url, dir=path, unzip=args.unzip, delete=args.delete, threads=args.threads)

def main():
    args = parse_args()
    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    ori_shape = (3, h, w)
    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor
    input_shape = (3, h, w)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError('FLOPs counter is currently not currently supported with {}'.format(model.__class__.__name__))
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    if divisor > 0 and input_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape from {ori_shape} to {input_shape}\n')
    print(f'{split_line}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. You may need to check if all ops are supported and verify that the flops computation is correct.')

def load_json_logs(json_logs):
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for i, line in enumerate(log_file):
                log = json.loads(line.strip())
                if i == 0:
                    continue
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts

def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)
    eval(args.task)(log_dicts, args)


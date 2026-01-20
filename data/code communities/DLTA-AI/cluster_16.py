# Cluster 16

def main(args):
    config = Config.fromfile(args.config)
    if args.model_name:
        if args.model_name in config:
            model_infos = config[args.model_name]
            if not isinstance(model_infos, list):
                model_infos = [model_infos]
            model_info = model_infos[0]
            config_name = model_info['config'].strip()
            print(f'processing: {config_name}', flush=True)
            checkpoint = osp.join(args.checkpoint_root, model_info['checkpoint'].strip())
            inference_model(config_name, checkpoint, args)
            return
        else:
            raise RuntimeError('model name input error.')
    logger = get_root_logger(log_file='benchmark_test_image.log', log_level=logging.ERROR)
    for model_key in config:
        model_infos = config[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print('processing: ', model_info['config'], flush=True)
            config_name = model_info['config'].strip()
            checkpoint = osp.join(args.checkpoint_root, model_info['checkpoint'].strip())
            try:
                inference_model(config_name, checkpoint, args, logger)
            except Exception as e:
                logger.error(f'{config_name} " : {repr(e)}')

def inference_model(config_name, checkpoint, args, logger=None):
    cfg = Config.fromfile(config_name)
    if args.aug:
        if 'flip' in cfg.data.test.pipeline[1]:
            cfg.data.test.pipeline[1].flip = True
        elif logger is not None:
            logger.error(f'{config_name}: unable to start aug test')
        else:
            print(f'{config_name}: unable to start aug test', flush=True)
    model = init_detector(cfg, checkpoint, device=args.device)
    result = inference_detector(model, args.img)
    if args.show:
        show_result_pyplot(model, args.img, result, score_thr=args.score_thr, wait_time=args.wait_time)
    return result


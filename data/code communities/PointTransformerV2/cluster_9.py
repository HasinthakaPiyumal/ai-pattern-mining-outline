# Cluster 9

def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(f'logger should be either a logging.Logger object, str, "silent" or None, but got {type(logger)}')

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='a', color=False):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'a'.
        color (bool): Colorful log output. Defaults to True

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    plain_formatter = logging.Formatter('[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s')
    if color:
        formatter = _ColorfulFormatter(colored('[%(asctime)s %(name)s]: ', 'green') + '%(message)s', datefmt='%m/%d %H:%M:%S', root_name=name)
    else:
        formatter = plain_formatter
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    return logger

def get_root_logger(log_file=None, log_level=logging.INFO, file_mode='a'):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "lseg3d".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
        file_mode (str): File Mode of logger. (w or a)

    Returns:
        logging.Logger: The root logger.
    """
    logger = get_logger(name='pcr', log_file=log_file, log_level=log_level, file_mode=file_mode)
    return logger

class Trainer:

    def __init__(self, cfg):
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.eval_metric = cfg.eval_metric
        self.best_metric_value = -torch.inf
        self.iter_end_time = None
        self.max_iter = None
        self.logger = get_root_logger(log_file=os.path.join(cfg.save_path, 'train.log'), file_mode='a' if cfg.resume else 'w')
        self.logger.info('=> Loading config ...')
        self.cfg = cfg
        self.logger.info(f'Save path: {cfg.save_path}')
        self.logger.info(f'Config:\n{cfg.pretty_text}')
        self.storage: EventStorage
        self.logger.info('=> Building model ...')
        self.model = self.build_model()
        self.logger.info('=> Building writer ...')
        self.writer = self.build_writer()
        self.logger.info('=> Building train dataset & dataloader ...')
        self.train_loader = self.build_train_loader()
        self.logger.info('=> Building val dataset & dataloader ...')
        self.val_loader = self.build_val_loader()
        self.logger.info('=> Building criteria, optimize, scheduler, scaler(amp) ...')
        self.criteria = self.build_criteria()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info('=> Checking load & resume ...')
        self.resume_or_load()

    def train(self):
        with EventStorage() as self.storage:
            self.logger.info('>>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>')
            self.max_iter = self.max_epoch * len(self.train_loader)
            for self.epoch in range(self.start_epoch, self.max_epoch):
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.start_epoch)
                self.model.train()
                self.iter_end_time = time.time()
                for i, input_dict in enumerate(self.train_loader):
                    self.run_step(i, input_dict)
                self.after_epoch()
            self.logger.info('==>Training done!\nBest {}: {:.4f}'.format(self.cfg.eval_metric, self.best_metric_value))
            if self.writer is not None:
                self.writer.close()

    def run_step(self, i, input_dict):
        data_time = time.time() - self.iter_end_time
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output = self.model(input_dict)
            loss = self.criteria(output, input_dict['label'])
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        n = input_dict['coord'].size(0)
        if comm.get_world_size() > 1:
            loss *= n
            count = input_dict['label'].new_tensor([n], dtype=torch.long)
            (dist.all_reduce(loss), dist.all_reduce(count))
            n = count.item()
            loss /= n
        batch_time = time.time() - self.iter_end_time
        self.iter_end_time = time.time()
        self.storage.put_scalar('loss', loss.item(), n=n)
        self.storage.put_scalar('data_time', data_time)
        self.storage.put_scalar('batch_time', batch_time)
        current_iter = self.epoch * len(self.train_loader) + i + 1
        remain_iter = self.max_iter - current_iter
        remain_time = remain_iter * self.storage.history('batch_time').avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        self.logger.info('Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] Scan {batch_size} ({points_num}) Data {data_time_val:.3f} ({data_time_avg:.3f}) Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) Remain {remain_time} Lr {lr:.4f} Loss {loss:.4f} '.format(epoch=self.epoch + 1, max_epoch=self.max_epoch, iter=i + 1, max_iter=len(self.train_loader), batch_size=len(input_dict['offset']), points_num=input_dict['offset'][-1], data_time_val=data_time, data_time_avg=self.storage.history('data_time').avg, batch_time_val=batch_time, batch_time_avg=self.storage.history('batch_time').avg, remain_time=remain_time, lr=self.optimizer.state_dict()['param_groups'][0]['lr'], loss=loss.item()))
        if i == 0:
            self.storage.history('data_time').reset()
            self.storage.history('batch_time').reset()
        if self.writer is not None:
            self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], current_iter)
            self.writer.add_scalar('train_batch/loss', loss.item(), current_iter)

    def after_epoch(self):
        loss_avg = self.storage.history('loss').avg
        self.logger.info('Train result: loss {:.4f}.'.format(loss_avg))
        current_epoch = self.epoch + 1
        if self.writer is not None:
            self.writer.add_scalar('train/loss', loss_avg, current_epoch)
        self.storage.reset_histories()
        if self.cfg.evaluate:
            self.eval()
        self.save_checkpoint()
        self.storage.reset_histories()

    def eval(self):
        self.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.model.eval()
        self.iter_end_time = time.time()
        for i, input_dict in enumerate(self.val_loader):
            data_time = time.time() - self.iter_end_time
            for key in input_dict.keys():
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(input_dict)
            loss = self.criteria(output, input_dict['label'].long())
            n = input_dict['coord'].size(0)
            if comm.get_world_size() > 1:
                loss *= n
                count = input_dict['label'].new_tensor([n], dtype=torch.long)
                (dist.all_reduce(loss), dist.all_reduce(count))
                n = count.item()
                loss /= n
            pred = output.max(1)[1]
            label = input_dict['label']
            if 'origin_coord' in input_dict.keys():
                idx, _ = pointops.knn_query(1, input_dict['coord'].float(), input_dict['offset'].int(), input_dict['origin_coord'].float(), input_dict['origin_offset'].int())
                pred = pred[idx.flatten().long()]
                label = input_dict['origin_label']
            intersection, union, target = intersection_and_union_gpu(pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_label)
            if comm.get_world_size() > 1:
                (dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target))
            intersection, union, target = (intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy())
            batch_time = time.time() - self.iter_end_time
            self.iter_end_time = time.time()
            self.storage.put_scalar('intersection', intersection)
            self.storage.put_scalar('union', union)
            self.storage.put_scalar('target', target)
            self.storage.put_scalar('loss', loss.item(), n=n)
            self.storage.put_scalar('data_time', data_time)
            self.storage.put_scalar('batch_time', batch_time)
            self.logger.info('Test: [{iter}/{max_iter}] Data {data_time_val:.3f} ({data_time_avg:.3f}) Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) Loss {loss:.4f} '.format(iter=i + 1, max_iter=len(self.val_loader), data_time_val=data_time, data_time_avg=self.storage.history('data_time').avg, batch_time_val=batch_time, batch_time_avg=self.storage.history('batch_time').avg, loss=loss.item()))
        loss_avg = self.storage.history('loss').avg
        intersection = self.storage.history('intersection').total
        union = self.storage.history('union').total
        target = self.storage.history('target').total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.storage.put_scalar('mIoU', m_iou)
        self.storage.put_scalar('mAcc', m_acc)
        self.storage.put_scalar('allAcc', all_acc)
        self.logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(m_iou, m_acc, all_acc))
        for i in range(self.cfg.data.num_classes):
            self.logger.info('Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(idx=i, name=self.cfg.data.names[i], iou=iou_class[i], accuracy=acc_class[i]))
        current_epoch = self.epoch + 1
        if self.writer is not None:
            self.writer.add_scalar('val/loss', loss_avg, current_epoch)
            self.writer.add_scalar('val/mIoU', m_iou, current_epoch)
            self.writer.add_scalar('val/mAcc', m_acc, current_epoch)
            self.writer.add_scalar('val/allAcc', all_acc, current_epoch)
        self.logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    def save_checkpoint(self):
        if comm.is_main_process():
            is_best = False
            current_metric_value = self.storage.latest()[self.cfg.eval_metric][0] if self.cfg.evaluate else 0
            if self.cfg.evaluate and current_metric_value > self.best_metric_value:
                self.best_metric_value = current_metric_value
                is_best = True
            filename = os.path.join(self.cfg.save_path, 'model', 'model_last.pth')
            self.logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': self.epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'scaler': self.scaler.state_dict() if self.cfg.enable_amp else None, 'best_metric_value': self.best_metric_value}, filename + '.tmp')
            os.replace(filename + '.tmp', filename)
            if is_best:
                shutil.copyfile(filename, os.path.join(self.cfg.save_path, 'model', 'model_best.pth'))
                self.logger.info('Best validation {} updated to: {:.4f}'.format(self.cfg.eval_metric, self.best_metric_value))
            self.logger.info('Currently Best {}: {:.4f}'.format(self.cfg.eval_metric, self.best_metric_value))
            if self.cfg.save_freq and self.cfg.save_freq % (self.epoch + 1) == 0:
                shutil.copyfile(filename, os.path.join(self.cfg.save_path, 'model', f'epoch_{self.epoch + 1}.pth'))

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum((p.numel() for p in model.parameters() if p.requires_grad))
        self.logger.info(f'Num params: {n_parameters}')
        model = create_ddp_model(model.cuda(), broadcast_buffers=False, find_unused_parameters=self.cfg.find_unused_parameters)
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None
        init_fn = partial(worker_init_fn, num_workers=self.cfg.num_worker_per_gpu, rank=comm.get_rank(), seed=self.cfg.seed) if self.cfg.seed is not None else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.cfg.batch_size_per_gpu, shuffle=train_sampler is None, num_workers=self.cfg.num_worker_per_gpu, sampler=train_sampler, collate_fn=partial(point_collate_fn, max_batch_points=self.cfg.max_batch_points, mix_prob=self.cfg.mix_prob), pin_memory=True, worker_init_fn=init_fn, drop_last=True, persistent_workers=True)
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.cfg.batch_size_val_per_gpu, shuffle=False, num_workers=self.cfg.num_worker_per_gpu, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)
        return val_loader

    def build_criteria(self):
        return build_criteria(self.cfg.criteria)

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, 'optimizer')
        assert hasattr(self, 'train_loader')
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler

    def resume_or_load(self):
        if self.cfg.weight and os.path.isfile(self.cfg.weight):
            self.logger.info(f'Loading weight at: {self.cfg.weight}')
            checkpoint = torch.load(self.cfg.weight, map_location=lambda storage, loc: storage.cuda())
            load_state_info = self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info(f'Missing keys: {load_state_info[0]}')
            if self.cfg.resume:
                self.logger.info(f'Resuming train at eval epoch: {checkpoint['epoch']}')
                self.start_epoch = checkpoint['epoch']
                self.best_metric_value = checkpoint['best_metric_value']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                if self.cfg.enable_amp:
                    self.scaler.load_state_dict(checkpoint['scaler'])
        else:
            self.logger.info(f'No weight found at: {self.cfg.weight}')

@TEST.register_module()
class SegmentationTest(object):
    """SegmentationTest
    for large outdoor point cloud with need voxelize (s3dis)
    """

    def __call__(self, cfg, test_loader, model):
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        model.eval()
        save_path = os.path.join(cfg.save_path, 'result', 'test_epoch{}'.format(cfg.epochs))
        make_dirs(save_path)
        if 'ScanNet' in cfg.dataset_type:
            sub_path = os.path.join(save_path, 'submit')
            make_dirs(sub_path)
        pred_save, label_save = ([], [])
        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)
            pred_save_path = os.path.join(save_path, '{}_pred.npy'.format(data_name))
            label_save_path = os.path.join(save_path, '{}_label.npy'.format(data_name))
            if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
                logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(test_dataset), data_name))
                pred, label = (np.load(pred_save_path), np.load(label_save_path))
            else:
                data_dict_list, label = test_dataset[idx]
                pred = torch.zeros((label.size, cfg.data.num_classes)).cuda()
                batch_num = int(np.ceil(len(data_dict_list) / cfg.batch_size_test))
                for i in range(batch_num):
                    s_i, e_i = (i * cfg.batch_size_test, min((i + 1) * cfg.batch_size_test, len(data_dict_list)))
                    input_dict = collate_fn(data_dict_list[s_i:e_i])
                    for key in input_dict.keys():
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict['index']
                    with torch.no_grad():
                        pred_part = model(input_dict)
                        pred_part = F.softmax(pred_part, -1)
                    if cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict['offset']:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be
                    logger.info('Test: {} {}/{}, Batch: {batch_idx}/{batch_num}'.format(data_name, idx + 1, len(test_dataset), batch_idx=i, batch_num=batch_num))
                pred = pred.max(1)[1].data.cpu().numpy()
            intersection, union, target = intersection_and_union(pred, label, cfg.data.num_classes, cfg.data.ignore_label)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}]-{} Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) Accuracy {acc:.4f} ({m_acc:.4f}) mIoU {iou:.4f} ({m_iou:.4f})'.format(data_name, idx + 1, len(test_dataset), label.size, batch_time=batch_time, acc=acc, m_acc=m_acc, iou=iou, m_iou=m_iou))
            pred_save.append(pred)
            label_save.append(label)
            np.save(pred_save_path, pred)
            np.save(label_save_path, label)
            if 'ScanNet' in cfg.dataset_type:
                np.savetxt(os.path.join(save_path, 'submit', '{}.txt'.format(data_name)), test_dataset.class2id[pred].reshape([-1, 1]), fmt='%d')
        with open(os.path.join(save_path, 'pred.pickle'), 'wb') as handle:
            pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_path, 'label.pickle'), 'wb') as handle:
            pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.data.num_classes):
            logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(idx=i, name=cfg.data.names[i], iou=iou_class[i], accuracy=accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def intersection_and_union(output, target, K, ignore_index=255):
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return (area_intersection, area_union, area_target)

@TEST.register_module()
class PartSegmentationTest(object):
    """PartSegmentationTest
    """

    def __call__(self, cfg, test_loader, model):
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        num_categories = len(test_loader.dataset.categories)
        iou_category, iou_count = (np.zeros(num_categories), np.zeros(num_categories))
        model.eval()
        save_path = os.path.join(cfg.save_path, 'result', 'test_epoch{}'.format(cfg.epochs))
        make_dirs(save_path)
        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)
            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = (i * cfg.batch_size_test, min((i + 1) * cfg.batch_size_test, len(data_dict_list)))
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = model(input_dict)
                    pred_part = F.softmax(pred_part, -1)
                if cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info('Test: {} {}/{}, Batch: {batch_idx}/{batch_num}'.format(data_name, idx + 1, len(test_dataset), batch_idx=i, batch_num=batch_num))
            pred = pred.max(1)[1].data.cpu().numpy()
            category_index = data_dict_list[0]['cls_token']
            category = test_loader.dataset.categories[category_index]
            parts_idx = test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if np.sum(label == part) == 0 and np.sum(pred == part) == 0:
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1
            batch_time.update(time.time() - end)
            logger.info('Test: {} [{}/{}] Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(data_name, idx + 1, len(test_loader), batch_time=batch_time))
        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info('Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.'.format(ins_mIoU, cat_mIoU))
        for i in range(num_categories):
            logger.info('Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}'.format(idx=i, name=test_loader.dataset.categories[i], iou_cat=iou_category[i] / (iou_count[i] + 1e-10), iou_count=int(iou_count[i])))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

@DATASETS.register_module()
class ScanNetPairDataset(Dataset):

    def __init__(self, data_root='data/scannet_pair', overlap_threshold=0.3, twin1_transform=None, twin2_transform=None, loop=1, **kwargs):
        super(ScanNetPairDataset, self).__init__()
        self.data_root = data_root
        self.overlap_threshold = overlap_threshold
        self.twin1_transform = Compose(twin1_transform)
        self.twin2_transform = Compose(twin2_transform)
        self.loop = loop
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info('Totally {} x {} samples.'.format(len(self.data_list), self.loop))

    def get_data_list(self):
        data_list = []
        overlap_list = glob.glob(os.path.join(self.data_root, '*', 'pcd', 'overlap.txt'))
        for overlap_file in overlap_list:
            with open(overlap_file) as f:
                overlap = f.readlines()
            overlap = [pair.strip().split() for pair in overlap]
            data_list.extend([pair[:2] for pair in overlap if float(pair[2]) > self.overlap_threshold])
        return data_list

    def get_data(self, idx):
        pair = self.data_list[idx % len(self.data_list)]
        twin1_dict = torch.load(self.data_root + pair[0])
        twin2_dict = torch.load(self.data_root + pair[1])
        twin1_dict['origin_coord'] = twin1_dict['coord'].copy()
        twin2_dict['origin_coord'] = twin2_dict['coord'].copy()
        return (twin1_dict, twin2_dict)

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split('.')[0]

    def prepare_train_data(self, idx):
        twin1_dict, twin2_dict = self.get_data(idx)
        twin1_dict = self.twin1_transform(twin1_dict)
        twin2_dict = self.twin2_transform(twin2_dict)
        data_dict = dict()
        for key, value in twin1_dict.items():
            data_dict['twin1_' + key] = value
        for key, value in twin2_dict.items():
            data_dict['twin2_' + key] = value
        return data_dict

    def prepare_test_data(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

@DATASETS.register_module()
class SemanticKITTIDataset(Dataset):

    def __init__(self, split='train', data_root='data/semantic_kitti', learning_map=None, transform=None, test_mode=False, test_cfg=None, loop=1):
        super(SemanticKITTIDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.learning_map = learning_map
        self.split2seq = dict(train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10], val=[8], test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop)
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]
        if isinstance(self.split, str):
            seq_list = self.split2seq[split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += self.split2seq[split]
        else:
            raise NotImplementedError
        self.data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, 'sequences', seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, 'velodyne')))
            self.data_list += [os.path.join(seq_folder, 'velodyne', file) for file in seq_files]
        logger = get_root_logger()
        logger.info('Totally {} x {} samples in {} set.'.format(len(self.data_list), self.loop, split))

    def prepare_train_data(self, idx):
        data_idx = idx % len(self.data_list)
        with open(self.data_list[data_idx], 'rb') as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])
        label_file = self.data_list[data_idx].replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                label = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            label = np.zeros(coord.shape[0]).astype(np.int32)
        label = np.vectorize(self.learning_map.__getitem__)(label & 65535).astype(np.int64)
        data_dict = dict(coord=coord, strength=strength, label=label)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        raise NotImplementedError

    def get_data_name(self, idx):
        return self.data_list[self.data_list[idx % len(self.data_list)]]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

@DATASETS.register_module()
class ArkitScenesDataset(Dataset):

    def __init__(self, split='Training', data_root='data/ARKitScenesMesh', transform=None, test_mode=False, test_cfg=None, loop=1):
        super(ArkitScenesDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.class2id = np.array(VALID_CLASS_IDS_200)
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop)
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info('Totally {} x {} samples in {} set.'.format(len(self.data_list), self.loop, split))

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, '*.pth'))
        elif isinstance(self.split, list):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, '*.pth'))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data['coord']
        color = data['color']
        normal = data['normal']
        label = np.zeros(coord.shape[0])
        data_dict = dict(coord=coord, normal=normal, color=color, label=label)
        return data_dict

    def get_data_name(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        return os.path.basename(self.data_list[data_idx]).split('.')[0]

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        label = data_dict.pop('label')
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                data_part_list = self.test_crop(data_part)
                input_dict_list += data_part_list
        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        return (input_dict_list, label)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

@DATASETS.register_module()
class DefaultDataset(Dataset):

    def __init__(self, split='train', data_root='data/dataset', transform=None, test_mode=False, test_cfg=None, loop=1):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop)
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info('Totally {} x {} samples in {} set.'.format(len(self.data_list), self.loop, split))

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, '*.pth'))
        elif isinstance(self.split, list):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, '*.pth'))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data['coord']
        color = data['color']
        normal = data['normal']
        if 'semantic_gt' in data.keys():
            label = data['semantic_gt'].reshape([-1])
        else:
            label = np.zeros(coord.shape[0])
        data_dict = dict(coord=coord, norm=normal, color=color, label=label)
        return data_dict

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return os.path.basename(self.data_list[data_idx]).split('.')[0]

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        label = data_dict.pop('label')
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part
        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        return (input_dict_list, label)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

@DATASETS.register_module()
class ConcatDataset(Dataset):

    def __init__(self, datasets, loop=1):
        super(ConcatDataset, self).__init__()
        self.datasets = [build_dataset(dataset) for dataset in datasets]
        self.loop = loop
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info('Totally {} x {} samples in the concat set.'.format(len(self.data_list), self.loop))

    def get_data_list(self):
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(zip(np.ones(len(self.datasets[i]), dtype=np.long) * i, np.arange(len(self.datasets[i]))))
        return data_list

    def get_data(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx][data_idx]

    def get_data_name(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx].get_data_name(data_idx)

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

@DATASETS.register_module()
class S3DISDataset(Dataset):

    def __init__(self, split=('Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'), data_root='data/s3dis', transform=None, test_mode=False, test_cfg=None, loop=1):
        super(S3DISDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info('Totally {} x {} samples in {} set.'.format(len(self.data_list), self.loop, split))

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, '*.pth'))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, '*.pth'))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data['coord']
        color = data['color']
        if 'semantic_gt' in data.keys():
            label = data['semantic_gt'].reshape([-1])
        else:
            label = np.zeros(coord.shape[0])
        data_dict = dict(coord=coord, color=color, label=label)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split('.')[0]

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        label = data_dict.pop('label')
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part
        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        return (input_dict_list, label)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

@DATASETS.register_module()
class ModelNetDataset(Dataset):

    def __init__(self, split='train', data_root='data/modelnet40_normal_resampled', class_names=None, transform=None, cache_data=False, test_mode=False, test_cfg=None, loop=1):
        super(ModelNetDataset, self).__init__()
        self.data_root = data_root
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.cache_data = cache_data
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}
        if test_mode:
            pass
        self.data_list = [line.rstrip() for line in open(os.path.join(self.data_root, 'modelnet40_{}.txt'.format(self.split)))]
        logger = get_root_logger()
        logger.info('Totally {} x {} samples in {} set.'.format(len(self.data_idx), self.loop, split))

    def prepare_train_data(self, idx):
        data_idx = idx % len(self.data_list)
        if self.cache_data:
            coord, norm, label = self.cache[data_idx]
        else:
            data_shape = '_'.join(self.data_list[data_idx].split('_')[0:-1])
            data_path = os.path.join(self.data_root, data_shape, self.data_list[data_idx] + '.txt')
            data = np.loadtxt(data_path, delimiter=',').astype(np.float32)
            coord, norm = (data[:, 0:3], data[:, 3:6])
            label = np.array([self.class_names[data_shape]])
            if self.cache_data:
                self.cache[data_idx] = (coord, norm, label)
        data_dict = dict(coord=coord, norm=norm, label=label)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self.data_idx)
        data_idx = idx
        data_shape = '_'.join(self.data_list[data_idx].split('_')[0:-1])
        data_path = os.path.join(self.data_root, data_shape, self.data_list[data_idx] + '.txt')
        data = np.loadtxt(data_path, delimiter=',').astype(np.float32)
        coord, norm = (data[:, 0:3], data[:, 3:6])
        label = np.array([self.class_names[data_shape]])
        data_dict = dict(coord=coord, norm=norm, label=label)
        data_dict = self.transform(data_dict)
        return data_dict

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_idx) * self.loop

@DATASETS.register_module()
class ShapeNetPartDataset(Dataset):

    def __init__(self, split='train', data_root='data/shapenetcore_partanno_segmentation_benchmark_v0_normal', transform=None, test_mode=False, test_cfg=None, loop=1):
        super(ShapeNetPartDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.cache = {}
        self.categories = []
        self.category2part = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11], 'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21], 'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40], 'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
        self.token2category = {}
        with open(os.path.join(self.data_root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.token2category[ls[1]] = len(self.categories)
                self.categories.append(ls[0])
        if test_mode:
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]
        if isinstance(self.split, str):
            self.data_list = self.load_data_list(self.split)
        elif isinstance(self.split, list):
            self.data_list = []
            for s in self.split:
                self.data_list += self.load_data_list(s)
        else:
            raise NotImplementedError
        logger = get_root_logger()
        logger.info('Totally {} x {} samples in {} set.'.format(len(self.data_idx), self.loop, split))

    def load_data_list(self, split):
        split_file = os.path.join(self.data_root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        if not os.path.isfile(split_file):
            raise RuntimeError('Split file do not exist: ' + split_file + '\n')
        with open(split_file, 'r') as f:
            data_list = [os.path.join(self.data_root, data[11:] + '.txt') for data in json.load(f)]
        return data_list

    def prepare_train_data(self, idx):
        data_idx = idx % len(self.data_list)
        if data_idx in self.cache:
            coord, norm, label, cls_token = self.cache[data_idx]
        else:
            data = np.loadtxt(self.data_list[data_idx]).astype(np.float32)
            cls_token = self.token2category[os.path.basename(os.path.dirname(self.data_list[data_idx]))]
            coord, norm, label = (data[:, :3], data[:, 3:6], data[:, 6].astype(np.int32))
            self.cache[data_idx] = (coord, norm, label, cls_token)
        data_dict = dict(coord=coord, norm=norm, label=label, cls_token=cls_token)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = np.loadtxt(self.data_list[data_idx]).astype(np.float32)
        cls_token = self.token2category[os.path.basename(os.path.dirname(self.data_list[data_idx]))]
        coord, norm, label = (data[:, :3], data[:, 3:6], data[:, 6].astype(np.int32))
        data_dict = dict(coord=coord, norm=norm, cls_token=cls_token)
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(self.post_transform(aug(deepcopy(data_dict))))
        return (data_dict_list, label)

    def get_data_name(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        return os.path.basename(self.data_list[data_idx]).split('.')[0]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_idx) * self.loop

@DATASETS.register_module()
class ScanNetDataset(Dataset):
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(self, split='train', data_root='data/scannet', transform=None, test_mode=False, test_cfg=None, loop=1):
        super(ScanNetDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info('Totally {} x {} samples in {} set.'.format(len(self.data_list), self.loop, split))

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, '*.pth'))
        elif isinstance(self.split, list):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, '*.pth'))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data['coord']
        color = data['color']
        normal = data['normal']
        if 'semantic_gt20' in data.keys():
            label = data['semantic_gt20'].reshape([-1])
        else:
            label = np.ones(coord.shape[0]) * 255
        data_dict = dict(coord=coord, normal=normal, color=color, label=label)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split('.')[0]

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        label = data_dict.pop('label')
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part
        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        return (input_dict_list, label)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


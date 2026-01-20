# Cluster 30

def de_parallel(model):
    return model.module if is_parallel(model) else model

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)

class ComputeLoss:

    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device
        h = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = (FocalLoss(BCEcls, g), FocalLoss(BCEobj, g))
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (BCEcls, BCEobj, 1.0, h, autobalance)
        for k in ('na', 'nc', 'nl', 'anchors'):
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = (torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device))
        tcls, tbox, indices, anchors = self.build_targets(p, targets)
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = (b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id])
                tobj[b, a, gj, gi] = 1.0 - self.gr + self.gr * score_iou
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]
        return ((lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach())

    def build_targets(self, p, targets):
        na, nt = (self.na, targets.shape[0])
        tcls, tbox, indices, anch = ([], [], [], [])
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        return (tcls, tbox, indices, anch)

def smooth_BCE(eps=0.1):
    return (1.0 - 0.5 * eps, 0.5 * eps)

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


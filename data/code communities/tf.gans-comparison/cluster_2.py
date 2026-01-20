# Cluster 2

def train(model, dataset, input_op, num_epochs, batch_size, n_examples, ckpt_step, renew=False):
    print('\n# of examples: {}'.format(n_examples))
    print('steps per epoch: {}\n'.format(n_examples // batch_size))
    summary_path = os.path.join('./summary/', dataset, model.name)
    ckpt_path = os.path.join('./checkpoints', dataset, model.name)
    if renew:
        if os.path.exists(summary_path):
            tf.gfile.DeleteRecursively(summary_path)
        if os.path.exists(ckpt_path):
            tf.gfile.DeleteRecursively(ckpt_path)
    if not os.path.exists(ckpt_path):
        tf.gfile.MakeDirs(ckpt_path)
    config = tf.ConfigProto()
    best_gpu = utils.get_best_gpu()
    config.gpu_options.visible_device_list = str(best_gpu)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        total_steps = int(np.ceil(n_examples * num_epochs / float(batch_size)))
        config_list = [('num_epochs', num_epochs), ('total_iteration', total_steps), ('batch_size', batch_size), ('dataset', dataset)]
        model_config_list = [[k, str(w)] for k, w in sorted(model.args.items()) + config_list]
        model_config_summary_op = tf.summary.text(model.name + '/config', tf.convert_to_tensor(model_config_list), collections=[])
        model_config_summary = sess.run(model_config_summary_op)
        print('\n====== Process info =======')
        print('argv: {}'.format(' '.join(sys.argv)))
        print('PID: {}'.format(os.getpid()))
        print('====== Model configs ======')
        for k, v in model_config_list:
            print('{}: {}'.format(k, v))
        print('===========================\n')
        summary_writer = tf.summary.FileWriter(summary_path, flush_secs=30, graph=sess.graph)
        summary_writer.add_summary(model_config_summary)
        pbar = tqdm(total=total_steps, desc='global_step')
        saver = tf.train.Saver(max_to_keep=9999)
        global_step = 0
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = sess.run(model.global_step)
            print('\n[!] Restore from {} ... starting global step is {}\n'.format(ckpt.model_checkpoint_path, global_step))
            pbar.update(global_step)
        try:
            while not coord.should_stop() and global_step < total_steps:
                summary_op = model.summary_op if global_step % 100 == 0 else model.all_summary_op
                batch_X = sess.run(input_op)
                batch_z = sample_z([batch_size, model.z_dim])
                _, summary = sess.run([model.D_train_op, summary_op], {model.X: batch_X, model.z: batch_z})
                _, global_step = sess.run([model.G_train_op, model.global_step], {model.z: batch_z})
                summary_writer.add_summary(summary, global_step=global_step)
                if global_step % 10 == 0:
                    pbar.update(10)
                    if global_step % ckpt_step == 0:
                        saver.save(sess, ckpt_path + '/' + model.name, global_step=global_step)
        except tf.errors.OutOfRangeError:
            print('\nDone -- epoch limit reached\n')
        finally:
            coord.request_stop()
        coord.join(threads)
        summary_writer.close()
        pbar.close()

def get_best_gpu():
    """Dependency: pynvml (for gpu memory informations)
    return type is integer (gpu_id)
    """
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo
    except Exception as e:
        print('[!] {} => Use default GPU settings ...\n'.format(e))
        return ''
    print('\n===== Check GPU memory =====')

    def to_mb(x):
        return int(x / 1024.0 / 1024.0)
    best_idx = -1
    best_free = 0.0
    nvmlInit()
    n_gpu = nvmlDeviceGetCount()
    for i in range(n_gpu):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)
        total = to_mb(mem.total)
        free = to_mb(mem.free)
        used = to_mb(mem.used)
        free_ratio = mem.free / float(mem.total)
        print('{} - {}/{} MB (free: {} MB - {:.2%})'.format(name, used, total, free, free_ratio))
        if free > best_free:
            best_free = free
            best_idx = i
    print('\nSelected GPU is gpu:{}'.format(best_idx))
    print('============================\n')
    return best_idx

def sample_z(shape):
    return np.random.normal(size=shape)

def to_mb(x):
    return int(x / 1024.0 / 1024.0)

def eval(model, name, dataset, sample_shape=[4, 4], load_all_ckpt=True):
    if name == None:
        name = model.name
    dir_name = os.path.join('eval', dataset, name)
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)
    restorer = tf.train.Saver(slim.get_model_variables())
    config = tf.ConfigProto()
    best_gpu = utils.get_best_gpu()
    config.gpu_options.visible_device_list = str(best_gpu)
    with tf.Session(config=config) as sess:
        ckpt_path = os.path.join('checkpoints', dataset, name)
        ckpts = get_all_checkpoints(ckpt_path, force=load_all_ckpt)
        size = sample_shape[0] * sample_shape[1]
        z_ = sample_z([size, model.z_dim])
        for v in ckpts:
            print('Evaluating {} ...'.format(v))
            restorer.restore(sess, v)
            global_step = int(v.split('/')[-1].split('-')[-1])
            fake_samples = sess.run(model.fake_sample, {model.z: z_})
            fake_samples = (fake_samples + 1.0) / 2.0
            merged_samples = utils.merge(fake_samples, size=sample_shape)
            fn = '{:0>6d}.png'.format(global_step)
            scipy.misc.imsave(os.path.join(dir_name, fn), merged_samples)

def get_all_checkpoints(ckpt_dir, force=False):
    """
    When the learning is interrupted and resumed, all checkpoints can not be fetched with get_checkpoint_state 
    (The checkpoint state is rewritten from the point of resume). 
    This function fetch all checkpoints forcely when arguments force=True.
    """
    if force:
        ckpts = os.listdir(ckpt_dir)
        ckpts = map(lambda p: os.path.splitext(p)[0], ckpts)
        ckpts = set(ckpts)
        ckpts = filter(lambda x: x.split('-')[-1].isdigit(), ckpts)
        ckpts = sorted(ckpts, key=lambda x: int(x.split('-')[-1]))
        ckpts = map(lambda x: os.path.join(ckpt_dir, x), ckpts)
    else:
        ckpts = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths
    return ckpts

def merge(images, size):
    """merge images - burrowed from @carpedm20.

    checklist before/after imsave:
    * are images post-processed? for example - denormalization
    * is np.squeeze required? maybe for grayscale...
    """
    h, w = (images.shape[1], images.shape[2])
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


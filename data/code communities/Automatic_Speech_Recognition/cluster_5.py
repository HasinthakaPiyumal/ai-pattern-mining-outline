# Cluster 5

class Runner(object):

    def _default_configs(self):
        return {'level': level, 'rnncell': rnncell, 'batch_size': batch_size, 'num_hidden': num_hidden, 'num_feature': num_feature, 'num_classes': num_classes, 'num_layer': num_layer, 'num_iter': num_iter, 'activation': activation_fn, 'optimizer': optimizer_fn, 'learning_rate': lr, 'keep_prob': keep_prob, 'grad_clip': grad_clip}

    @describe
    def load_data(self, args, mode, type):
        if mode == 'train':
            return load_batched_data(train_mfcc_dir, train_label_dir, batch_size, mode, type)
        elif mode == 'test':
            return load_batched_data(test_mfcc_dir, test_label_dir, batch_size, mode, type)
        else:
            raise TypeError('mode should be train or test.')

    def run(self):
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        batchedData, maxTimeSteps, totalN = self.load_data(args, mode=mode, type=level)
        model = model_fn(args, maxTimeSteps)
        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        print(model.config)
        with tf.Session() as sess:
            if keep == True:
                ckpt = tf.train.get_checkpoint_state(savedir)
                if ckpt and ckpt.model_checkpoint_path:
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Model restored from:' + savedir)
            else:
                print('Initializing')
                sess.run(model.initial_op)
            for epoch in range(num_epochs):
                start = time.time()
                if mode == 'train':
                    print('Epoch', epoch + 1, '...')
                batchErrors = np.zeros(len(batchedData))
                batchRandIxs = np.random.permutation(len(batchedData))
                for batch, batchOrigI in enumerate(batchRandIxs):
                    batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                    batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                    feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs, model.targetVals: batchTargetVals, model.targetShape: batchTargetShape, model.seqLengths: batchSeqLengths}
                    if level == 'cha':
                        if mode == 'train':
                            _, l, pre, y, er = sess.run([model.optimizer, model.loss, model.predictions, model.targetY, model.errorRate], feed_dict=feedDict)
                            batchErrors[batch] = er
                            print('\n{} mode, total:{},batch:{}/{},epoch:{}/{},train loss={:.3f},mean train CER={:.3f}\n'.format(level, totalN, batch + 1, len(batchRandIxs), epoch + 1, num_epochs, l, er / batch_size))
                        elif mode == 'test':
                            l, pre, y, er = sess.run([model.loss, model.predictions, model.targetY, model.errorRate], feed_dict=feedDict)
                            batchErrors[batch] = er
                            print('\n{} mode, total:{},batch:{}/{},test loss={:.3f},mean test CER={:.3f}\n'.format(level, totalN, batch + 1, len(batchRandIxs), l, er / batch_size))
                    elif level == 'phn':
                        if mode == 'train':
                            _, l, pre, y = sess.run([model.optimizer, model.loss, model.predictions, model.targetY], feed_dict=feedDict)
                            er = get_edit_distance([pre.values], [y.values], True, level)
                            print('\n{} mode, total:{},batch:{}/{},epoch:{}/{},train loss={:.3f},mean train PER={:.3f}\n'.format(level, totalN, batch + 1, len(batchRandIxs), epoch + 1, num_epochs, l, er))
                            batchErrors[batch] = er * len(batchSeqLengths)
                        elif mode == 'test':
                            l, pre, y = sess.run([model.loss, model.predictions, model.targetY], feed_dict=feedDict)
                            er = get_edit_distance([pre.values], [y.values], True, level)
                            print('\n{} mode, total:{},batch:{}/{},test loss={:.3f},mean test PER={:.3f}\n'.format(level, totalN, batch + 1, len(batchRandIxs), l, er))
                            batchErrors[batch] = er * len(batchSeqLengths)
                    if er / batch_size == 1.0:
                        break
                    if batch % 30 == 0:
                        print('Truth:\n' + output_to_sequence(y, type=level))
                        print('Output:\n' + output_to_sequence(pre, type=level))
                    if mode == 'train' and ((epoch * len(batchRandIxs) + batch + 1) % 20 == 0 or (epoch == num_epochs - 1 and batch == len(batchRandIxs) - 1)):
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in {}'.format(savedir))
                end = time.time()
                delta_time = end - start
                print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')
                if mode == 'train':
                    if (epoch + 1) % 1 == 0:
                        checkpoint_path = os.path.join(savedir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step=epoch)
                        print('Model has been saved in {}'.format(savedir))
                    epochER = batchErrors.sum() / totalN
                    print('Epoch', epoch + 1, 'mean train error rate:', epochER)
                    logging(model, logfile, epochER, epoch, delta_time, mode='config')
                    logging(model, logfile, epochER, epoch, delta_time, mode=mode)
                if mode == 'test':
                    with open(os.path.join(resultdir, level + '_result.txt'), 'a') as result:
                        result.write(output_to_sequence(y, type=level) + '\n')
                        result.write(output_to_sequence(pre, type=level) + '\n')
                        result.write('\n')
                    epochER = batchErrors.sum() / totalN
                    print(' test error rate:', epochER)
                    logging(model, logfile, epochER, mode=mode)

@describe
def count_params(model, mode='trainable'):
    """ count all parameters of a tensorflow graph
    """
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of ' + mode + ' parameters: ' + str(num))
    return num

def output_to_sequence(lmt, type='phn'):
    """ convert the output into sequences of characters or phonemes
    """
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    sequences = []
    start = 0
    sequences.append([])
    for i in range(len(lmt[0])):
        if lmt[0][i][0] == start:
            sequences[start].append(lmt[1][i])
        else:
            start = start + 1
            sequences.append([])
    indexes = sequences[0]
    if type == 'phn':
        seq = []
        for ind in indexes:
            if ind == len(phn):
                pass
            else:
                seq.append(phn[ind])
        seq = ' '.join(seq)
        return seq
    elif type == 'cha':
        seq = []
        for ind in indexes:
            if ind == 0:
                seq.append(' ')
            elif ind == 27:
                seq.append("'")
            elif ind == 28:
                pass
            else:
                seq.append(chr(ind + 96))
        seq = ''.join(seq)
        return seq
    else:
        raise TypeError('mode should be phoneme or character')

@describe
def logging(model, logfile, errorRate, epoch=0, delta_time=0, mode='train'):
    """ log the cost and error rate and time while training or testing
    """
    if mode != 'train' and mode != 'test' and (mode != 'config') and (mode != 'dev'):
        raise TypeError('mode should be train or test or config.')
    logfile = logfile
    if mode == 'config':
        with open(logfile, 'a') as myfile:
            myfile.write(str(model.config) + '\n')
    elif mode == 'train':
        with open(logfile, 'a') as myfile:
            myfile.write(str(time.strftime('%X %x %Z')) + '\n')
            myfile.write('Epoch:' + str(epoch + 1) + ' ' + 'train error rate:' + str(errorRate) + '\n')
            myfile.write('Epoch:' + str(epoch + 1) + ' ' + 'train time:' + str(delta_time) + ' s\n')
    elif mode == 'test':
        logfile = logfile + '_TEST'
        with open(logfile, 'a') as myfile:
            myfile.write(str(model.config) + '\n')
            myfile.write(str(time.strftime('%X %x %Z')) + '\n')
            myfile.write('test error rate:' + str(errorRate) + '\n')
    elif mode == 'dev':
        logfile = logfile + '_DEV'
        with open(logfile, 'a') as myfile:
            myfile.write(str(model.config) + '\n')
            myfile.write(str(time.strftime('%X %x %Z')) + '\n')
            myfile.write('development error rate:' + str(errorRate) + '\n')

class Runner(object):

    def _default_configs(self):
        return {'level': level, 'rnncell': rnncell, 'batch_size': batch_size, 'num_hidden': num_hidden, 'num_feature': num_feature, 'num_class': num_classes, 'num_layer': num_layer, 'activation': activation_fn, 'optimizer': optimizer_fn, 'learning_rate': lr, 'keep_prob': keep_prob, 'grad_clip': grad_clip}

    @describe
    def load_data(self, feature_dir, label_dir, mode, level):
        return load_batched_data(feature_dir, label_dir, batch_size, mode, level)

    def run(self):
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        feature_dirs, label_dirs = get_data(datadir, level, train_dataset, dev_dataset, test_dataset, mode)
        batchedData, maxTimeSteps, totalN = self.load_data(feature_dirs[0], label_dirs[0], mode, level)
        model = model_fn(args, maxTimeSteps)
        FL_pair = list(zip(feature_dirs, label_dirs))
        random.shuffle(FL_pair)
        feature_dirs, label_dirs = zip(*FL_pair)
        for feature_dir, label_dir in zip(feature_dirs, label_dirs):
            id_dir = feature_dirs.index(feature_dir)
            print('dir id:{}'.format(id_dir))
            batchedData, maxTimeSteps, totalN = self.load_data(feature_dir, label_dir, mode, level)
            model = model_fn(args, maxTimeSteps)
            num_params = count_params(model, mode='trainable')
            all_num_params = count_params(model, mode='all')
            model.config['trainable params'] = num_params
            model.config['all params'] = all_num_params
            print(model.config)
            with tf.Session(graph=model.graph) as sess:
                if keep == True:
                    ckpt = tf.train.get_checkpoint_state(savedir)
                    if ckpt and ckpt.model_checkpoint_path:
                        model.saver.restore(sess, ckpt.model_checkpoint_path)
                        print('Model restored from:' + savedir)
                else:
                    print('Initializing')
                    sess.run(model.initial_op)
                for epoch in range(num_epochs):
                    start = time.time()
                    if mode == 'train':
                        print('Epoch {} ...'.format(epoch + 1))
                    batchErrors = np.zeros(len(batchedData))
                    batchRandIxs = np.random.permutation(len(batchedData))
                    for batch, batchOrigI in enumerate(batchRandIxs):
                        batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                        batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                        feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs, model.targetVals: batchTargetVals, model.targetShape: batchTargetShape, model.seqLengths: batchSeqLengths}
                        if level == 'cha':
                            if mode == 'train':
                                _, l, pre, y, er = sess.run([model.optimizer, model.loss, model.predictions, model.targetY, model.errorRate], feed_dict=feedDict)
                                batchErrors[batch] = er
                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},epoch:{}/{},train loss={:.3f},mean train CER={:.3f}\n'.format(level, totalN, id_dir + 1, len(feature_dirs), batch + 1, len(batchRandIxs), epoch + 1, num_epochs, l, er / batch_size))
                            elif mode == 'dev':
                                l, pre, y, er = sess.run([model.loss, model.predictions, model.targetY, model.errorRate], feed_dict=feedDict)
                                batchErrors[batch] = er
                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},dev loss={:.3f},mean dev CER={:.3f}\n'.format(level, totalN, id_dir + 1, len(feature_dirs), batch + 1, len(batchRandIxs), l, er / batch_size))
                            elif mode == 'test':
                                l, pre, y, er = sess.run([model.loss, model.predictions, model.targetY, model.errorRate], feed_dict=feedDict)
                                batchErrors[batch] = er
                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},test loss={:.3f},mean test CER={:.3f}\n'.format(level, totalN, id_dir + 1, len(feature_dirs), batch + 1, len(batchRandIxs), l, er / batch_size))
                        elif level == 'seq2seq':
                            raise ValueError('level %s is not supported now' % str(level))
                        if er / batch_size == 1.0:
                            break
                        if batch % 20 == 0:
                            print('Truth:\n' + output_to_sequence(y, type=level))
                            print('Output:\n' + output_to_sequence(pre, type=level))
                        if mode == 'train' and ((epoch * len(batchRandIxs) + batch + 1) % 20 == 0 or (epoch == num_epochs - 1 and batch == len(batchRandIxs) - 1)):
                            checkpoint_path = os.path.join(savedir, 'model.ckpt')
                            model.saver.save(sess, checkpoint_path, global_step=epoch)
                            print('Model has been saved in {}'.format(savedir))
                    end = time.time()
                    delta_time = end - start
                    print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')
                    if mode == 'train':
                        if (epoch + 1) % 1 == 0:
                            checkpoint_path = os.path.join(savedir, 'model.ckpt')
                            model.saver.save(sess, checkpoint_path, global_step=epoch)
                            print('Model has been saved in {}'.format(savedir))
                        epochER = batchErrors.sum() / totalN
                        print('Epoch', epoch + 1, 'mean train error rate:', epochER)
                        logging(model, logfile, epochER, epoch, delta_time, mode='config')
                        logging(model, logfile, epochER, epoch, delta_time, mode=mode)
                    if mode == 'test' or mode == 'dev':
                        with open(os.path.join(resultdir, level + '_result.txt'), 'a') as result:
                            result.write(output_to_sequence(y, type=level) + '\n')
                            result.write(output_to_sequence(pre, type=level) + '\n')
                            result.write('\n')
                        epochER = batchErrors.sum() / totalN
                        print(' test error rate:', epochER)
                        logging(model, logfile, epochER, mode=mode)

def get_data(datadir, level, train_dataset, dev_dataset, test_dataset, mode):
    if mode == 'train':
        train_feature_dirs = [os.path.join(os.path.join(datadir, level, train_dataset), i, 'feature') for i in os.listdir(os.path.join(datadir, level, train_dataset))]
        train_label_dirs = [os.path.join(os.path.join(datadir, level, train_dataset), i, 'label') for i in os.listdir(os.path.join(datadir, level, train_dataset))]
        return (train_feature_dirs, train_label_dirs)
    if mode == 'dev':
        dev_feature_dirs = [os.path.join(os.path.join(datadir, level, dev_dataset), i, 'feature') for i in os.listdir(os.path.join(datadir, level, dev_dataset))]
        dev_label_dirs = [os.path.join(os.path.join(datadir, level, dev_dataset), i, 'label') for i in os.listdir(os.path.join(datadir, level, dev_dataset))]
        return (dev_feature_dirs, dev_label_dirs)
    if mode == 'test':
        test_feature_dirs = [os.path.join(os.path.join(datadir, level, test_dataset), i, 'feature') for i in os.listdir(os.path.join(datadir, level, test_dataset))]
        test_label_dirs = [os.path.join(os.path.join(datadir, level, test_dataset), i, 'label') for i in os.listdir(os.path.join(datadir, level, test_dataset))]
        return (test_feature_dirs, test_label_dirs)

class Runner(object):

    def _default_configs(self):
        return {'level': level, 'rnncell': rnncell, 'batch_size': batch_size, 'num_hidden': num_hidden, 'num_feature': num_feature, 'num_class': num_classes, 'num_layer': num_layer, 'activation': activation_fn, 'optimizer': optimizer_fn, 'learning_rate': lr, 'keep_prob': keep_prob, 'grad_clip': grad_clip}

    @describe
    def load_data(self, feature_dir, label_dir, mode, level):
        return load_batched_data(feature_dir, label_dir, batch_size, mode, level)

    def run(self):
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        feature_dirs, label_dirs = get_data(datadir, level, train_dataset, dev_dataset, test_dataset, mode)
        batchedData, maxTimeSteps, totalN = self.load_data(feature_dirs[0], label_dirs[0], mode, level)
        model = model_fn(args, maxTimeSteps)
        FL_pair = list(zip(feature_dirs, label_dirs))
        random.shuffle(FL_pair)
        feature_dirs, label_dirs = zip(*FL_pair)
        for feature_dir, label_dir in zip(feature_dirs, label_dirs):
            id_dir = feature_dirs.index(feature_dir)
            print('dir id:{}'.format(id_dir))
            batchedData, maxTimeSteps, totalN = self.load_data(feature_dir, label_dir, mode, level)
            model = model_fn(args, maxTimeSteps)
            num_params = count_params(model, mode='trainable')
            all_num_params = count_params(model, mode='all')
            model.config['trainable params'] = num_params
            model.config['all params'] = all_num_params
            print(model.config)
            with tf.Session(graph=model.graph) as sess:
                if keep == True:
                    ckpt = tf.train.get_checkpoint_state(savedir)
                    if ckpt and ckpt.model_checkpoint_path:
                        model.saver.restore(sess, ckpt.model_checkpoint_path)
                        print('Model restored from:' + savedir)
                else:
                    print('Initializing')
                    sess.run(model.initial_op)
                for epoch in range(num_epochs):
                    start = time.time()
                    if mode == 'train':
                        print('Epoch {} ...'.format(epoch + 1))
                    batchErrors = np.zeros(len(batchedData))
                    batchRandIxs = np.random.permutation(len(batchedData))
                    for batch, batchOrigI in enumerate(batchRandIxs):
                        batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                        batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                        feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs, model.targetVals: batchTargetVals, model.targetShape: batchTargetShape, model.seqLengths: batchSeqLengths}
                        if level == 'cha':
                            if mode == 'train':
                                _, l, pre, y, er = sess.run([model.optimizer, model.loss, model.predictions, model.targetY, model.errorRate], feed_dict=feedDict)
                                batchErrors[batch] = er
                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},epoch:{}/{},train loss={:.3f},mean train CER={:.3f}\n'.format(level, totalN, id_dir + 1, len(feature_dirs), batch + 1, len(batchRandIxs), epoch + 1, num_epochs, l, er / batch_size))
                            elif mode == 'dev':
                                l, pre, y, er = sess.run([model.loss, model.predictions, model.targetY, model.errorRate], feed_dict=feedDict)
                                batchErrors[batch] = er
                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},dev loss={:.3f},mean dev CER={:.3f}\n'.format(level, totalN, id_dir + 1, len(feature_dirs), batch + 1, len(batchRandIxs), l, er / batch_size))
                            elif mode == 'test':
                                l, pre, y, er = sess.run([model.loss, model.predictions, model.targetY, model.errorRate], feed_dict=feedDict)
                                batchErrors[batch] = er
                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},test loss={:.3f},mean test CER={:.3f}\n'.format(level, totalN, id_dir + 1, len(feature_dirs), batch + 1, len(batchRandIxs), l, er / batch_size))
                        elif level == 'seq2seq':
                            raise ValueError('level %s is not supported now' % str(level))
                        if er / batch_size == 1.0:
                            break
                        if batch % 20 == 0:
                            print('Truth:\n' + output_to_sequence(y, type=level))
                            print('Output:\n' + output_to_sequence(pre, type=level))
                        if mode == 'train' and ((epoch * len(batchRandIxs) + batch + 1) % 20 == 0 or (epoch == num_epochs - 1 and batch == len(batchRandIxs) - 1)):
                            checkpoint_path = os.path.join(savedir, 'model.ckpt')
                            model.saver.save(sess, checkpoint_path, global_step=epoch)
                            print('Model has been saved in {}'.format(savedir))
                    end = time.time()
                    delta_time = end - start
                    print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')
                    if mode == 'train':
                        if (epoch + 1) % 1 == 0:
                            checkpoint_path = os.path.join(savedir, 'model.ckpt')
                            model.saver.save(sess, checkpoint_path, global_step=epoch)
                            print('Model has been saved in {}'.format(savedir))
                        epochER = batchErrors.sum() / totalN
                        print('Epoch', epoch + 1, 'mean train error rate:', epochER)
                        logging(model, logfile, epochER, epoch, delta_time, mode='config')
                        logging(model, logfile, epochER, epoch, delta_time, mode=mode)
                    if mode == 'test' or mode == 'dev':
                        with open(os.path.join(resultdir, level + '_result.txt'), 'a') as result:
                            result.write(output_to_sequence(y, type=level) + '\n')
                            result.write(output_to_sequence(pre, type=level) + '\n')
                            result.write('\n')
                        epochER = batchErrors.sum() / totalN
                        print(' test error rate:', epochER)
                        logging(model, logfile, epochER, mode=mode)


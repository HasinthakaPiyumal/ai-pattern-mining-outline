# Cluster 1

def get_edit_distance(hyp_arr, truth_arr, normalize, level):
    """ calculate edit distance
    This is very universal, both for cha-level and phn-level
    """
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.reduce_sum(tf.edit_distance(hyp, truth, normalize=normalize))
    with tf.Session(graph=graph) as session:
        truthTest = list_to_sparse_tensor(truth_arr, level)
        hypTest = list_to_sparse_tensor(hyp_arr, level)
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run(editDist, feed_dict=feedDict)
    return dist

def list_to_sparse_tensor(targetList, level):
    """ turn 2-D List to SparseTensor
    """
    indices = []
    vals = []
    assert level == 'phn' or level == 'cha', 'type must be phoneme or character, seq2seq will be supported in future'
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', 'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', 'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#', 'q': 'h#', 'epi': 'h#', 'pau': 'h#'}
    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', 'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', 'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', 'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#', 'q': 'h#', 'epi': 'h#', 'pau': 'h#'}
    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
    if level == 'cha':
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(axis=0)[1] + 1]
        return (np.array(indices), np.array(vals), np.array(shape))
    elif level == 'phn':
        '\n        for phn level, we should collapse 61 labels into 39 labels before scoring\n        \n        Reference:\n          Heterogeneous Acoustic Measurements and Multiple Classifiers for Speech Recognition(1986), \n            Andrew K. Halberstadt, https://groups.csail.mit.edu/sls/publications/1998/phdthesis-drew.pdf\n        '
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                if val < len(phn) and phn[val] in mapping.keys():
                    val = group_phn.index(mapping[phn[val]])
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]
        return (np.array(indices), np.array(vals), np.array(shape))
    else:
        raise ValueError('Invalid level: %s' % str(level))

def data_lists_to_batches(inputList, targetList, batchSize, level):
    """ padding the input list to a same dimension, integrate all data into batchInputs
    """
    assert len(inputList) == len(targetList)
    nFeatures = inputList[0].shape[0]
    maxLength = 0
    for inp in inputList:
        maxLength = max(maxLength, inp.shape[1])
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []
    while end <= len(inputList):
        batchSeqLengths = np.zeros(batchSize)
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]
        batchInputs = np.zeros((maxLength, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            padSecs = maxLength - inputList[origI].shape[1]
            batchInputs[:, batchI, :] = np.pad(inputList[origI].T, ((0, padSecs), (0, 0)), 'constant', constant_values=0)
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, list_to_sparse_tensor(batchTargetList, level), batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxLength)

def load_batched_data(mfccPath, labelPath, batchSize, mode, level):
    """returns 3-element tuple: batched data (list), maxTimeLength (int), and
       total number of samples (int)"""
    return data_lists_to_batches([np.load(os.path.join(mfccPath, fn)) for fn in os.listdir(mfccPath)], [np.load(os.path.join(labelPath, fn)) for fn in os.listdir(labelPath)], batchSize, level) + (len(os.listdir(mfccPath)),)

def get_edit_distance(hyp_arr, truth_arr, mode='train'):
    """ calculate edit distance
    """
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.edit_distance(hyp, truth, normalize=True)
    with tf.Session(graph=graph) as session:
        truthTest = list_to_sparse_tensor(truth_arr, mode)
        hypTest = list_to_sparse_tensor(hyp_arr, mode)
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run(editDist, feed_dict=feedDict)
    return dist

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


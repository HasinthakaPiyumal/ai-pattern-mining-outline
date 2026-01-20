# Cluster 5

class DRAGAN(BaseModel):

    def __init__(self, name, training, D_lr=0.0001, G_lr=0.0001, image_shape=[64, 64, 3], z_dim=100):
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.ld = 10.0
        self.C = 0.5
        super(DRAGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            D_real_prob, D_real_logits = self._discriminator(X)
            D_fake_prob, D_fake_logits = self._discriminator(G, reuse=True)
            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake
            shape = tf.shape(X)
            eps = tf.random_uniform(shape=shape, minval=0.0, maxval=1.0)
            x_mean, x_var = tf.nn.moments(X, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)
            noise = self.C * x_std * eps
            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1.0, maxval=1.0)
            xhat = tf.clip_by_value(X + alpha * noise, -1.0, 1.0)
            D_xhat_prob, D_xhat_logits = self._discriminator(xhat, reuse=True)
            D_xhat_grad = tf.gradients(D_xhat_logits, xhat)[0]
            D_xhat_grad_norm = tf.norm(D_xhat_grad, axis=1)
            GP = self.ld * tf.reduce_mean(tf.square(D_xhat_grad_norm - 1.0))
            D_loss += GP
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/discriminator/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/generator/')
            D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1, beta2=self.beta2).minimize(D_loss, var_list=D_vars)
            G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1, beta2=self.beta2).minimize(G_loss, var_list=G_vars, global_step=global_step)
            self.summary_op = tf.summary.merge([tf.summary.scalar('G_loss', G_loss), tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('GP', GP)])
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.histogram('real_probs', D_real_prob)
            tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            net = X
            with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=2, activation_fn=ops.lrelu):
                net = slim.conv2d(net, 64)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])
            net = slim.flatten(net)
            logits = slim.fully_connected(net, 1, activation_fn=None)
            prob = tf.nn.sigmoid(logits)
            return (prob, logits)

    def _generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4 * 4 * 1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5, 5], stride=2, activation_fn=tf.nn.relu):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])
                return net

def expected_shape(tensor, expected):
    """batch size N shouldn't be set. 
    you can use shape of tensor instead of tensor itself.
    
    Usage:
    # batch size N is skipped.
    expected_shape(tensor, [28, 28, 1])
    expected_shape(tensor.shape, [28, 28, 1])
    """
    if isinstance(tensor, tf.Tensor):
        shape = tensor.shape[1:]
    else:
        shape = tensor[1:]
    shape = map(lambda x: x.value, shape)
    err_msg = 'wrong shape {} (expected shape is {})'.format(shape, expected)
    assert shape == expected, err_msg

class WGAN(BaseModel):

    def __init__(self, name, training, D_lr=5e-05, G_lr=5e-05, image_shape=[64, 64, 3], z_dim=100):
        self.ld = 10.0
        self.n_critic = 5
        super(WGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            C_real = self._critic(X)
            C_fake = self._critic(G, reuse=True)
            W_dist = tf.reduce_mean(C_real - C_fake)
            C_loss = -W_dist
            G_loss = tf.reduce_mean(-C_fake)
            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/generator/')
            C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/generator/')
            with tf.control_dependencies(C_update_ops):
                C_train_op = tf.train.RMSPropOptimizer(learning_rate=self.D_lr * self.n_critic).minimize(C_loss, var_list=C_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.RMSPropOptimizer(learning_rate=self.G_lr).minimize(G_loss, var_list=G_vars, global_step=global_step)
            ' It is right that clips gamma of the batch_norm? '
            C_clips = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in C_vars]
            with tf.control_dependencies([C_train_op]):
                C_train_op = tf.tuple(C_clips)
            self.summary_op = tf.summary.merge([tf.summary.scalar('G_loss', G_loss), tf.summary.scalar('C_loss', C_loss), tf.summary.scalar('W_dist', W_dist)])
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            self.all_summary_op = tf.summary.merge_all()
            self.X = X
            self.z = z
            self.D_train_op = C_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _critic(self, X, reuse=False):
        """ K-Lipschitz function """
        with tf.variable_scope('critic', reuse=reuse):
            net = X
            with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=2, activation_fn=ops.lrelu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d(net, 64, normalizer_fn=None)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, activation_fn=None)
            return net

    def _generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4 * 4 * 1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5, 5], stride=2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])
                return net

class WGAN_GP(BaseModel):

    def __init__(self, name, training, D_lr=0.0001, G_lr=0.0001, image_shape=[64, 64, 3], z_dim=100):
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.ld = 10.0
        self.n_critic = 5
        super(WGAN_GP, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            C_real = self._critic(X)
            C_fake = self._critic(G, reuse=True)
            W_dist = tf.reduce_mean(C_real - C_fake)
            C_loss = -W_dist
            G_loss = tf.reduce_mean(-C_fake)
            eps = tf.random_uniform(shape=[tf.shape(X)[0], 1, 1, 1], minval=0.0, maxval=1.0)
            x_hat = eps * X + (1.0 - eps) * G
            C_xhat = self._critic(x_hat, reuse=True)
            C_xhat_grad = tf.gradients(C_xhat, x_hat)[0]
            C_xhat_grad_norm = tf.norm(slim.flatten(C_xhat_grad), axis=1)
            GP = self.ld * tf.reduce_mean(tf.square(C_xhat_grad_norm - 1.0))
            C_loss += GP
            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/generator/')
            C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/generator/')
            n_critic = 5
            lr = 0.0001
            with tf.control_dependencies(C_update_ops):
                C_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr * n_critic, beta1=self.beta1, beta2=self.beta2).minimize(C_loss, var_list=C_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1, beta2=self.beta2).minimize(G_loss, var_list=G_vars, global_step=global_step)
            self.summary_op = tf.summary.merge([tf.summary.scalar('G_loss', G_loss), tf.summary.scalar('C_loss', C_loss), tf.summary.scalar('W_dist', W_dist), tf.summary.scalar('GP', GP)])
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            self.all_summary_op = tf.summary.merge_all()
            self.X = X
            self.z = z
            self.D_train_op = C_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _critic(self, X, reuse=False):
        return self._good_critic(X, reuse)

    def _generator(self, z, reuse=False):
        return self._good_generator(z, reuse)

    def _dcgan_critic(self, X, reuse=False):
        """
        K-Lipschitz function.
        WGAN-GP does not use critic in batch norm.
        """
        with tf.variable_scope('critic', reuse=reuse):
            net = X
            with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=ops.lrelu):
                net = slim.conv2d(net, 64)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, activation_fn=None)
            return net

    def _dcgan_generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4 * 4 * 1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5, 5], stride=2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])
                return net
    '\n    ResNet architecture from appendix C in the paper.\n    https://github.com/igul222/improved_wgan_training/blob/master/gan_64x64.py - GoodGenerator / GoodDiscriminator\n    layer norm in D, batch norm in G.\n    some details are ignored in this implemenation.\n    '

    def _residual_block(self, X, nf_output, resample, kernel_size=[3, 3], name='res_block'):
        with tf.variable_scope(name):
            input_shape = X.shape
            nf_input = input_shape[-1]
            if resample == 'down':
                shortcut = slim.avg_pool2d(X, [2, 2])
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1, 1], activation_fn=None)
                net = slim.layer_norm(X, activation_fn=tf.nn.relu)
                net = slim.conv2d(net, nf_input, kernel_size=kernel_size, biases_initializer=None)
                net = slim.layer_norm(net, activation_fn=tf.nn.relu)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)
                net = slim.avg_pool2d(net, [2, 2])
                return net + shortcut
            elif resample == 'up':
                upsample_shape = map(lambda x: int(x) * 2, input_shape[1:3])
                shortcut = tf.image.resize_nearest_neighbor(X, upsample_shape)
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1, 1], activation_fn=None)
                net = slim.batch_norm(X, activation_fn=tf.nn.relu, **self.bn_params)
                net = tf.image.resize_nearest_neighbor(net, upsample_shape)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size, biases_initializer=None)
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, **self.bn_params)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)
                return net + shortcut
            else:
                raise Exception('invalid resample value')

    def _good_generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            nf = 64
            net = slim.fully_connected(z, 4 * 4 * 8 * nf, activation_fn=None)
            net = tf.reshape(net, [-1, 4, 4, 8 * nf])
            net = self._residual_block(net, 8 * nf, resample='up', name='res_block1')
            net = self._residual_block(net, 4 * nf, resample='up', name='res_block2')
            net = self._residual_block(net, 2 * nf, resample='up', name='res_block3')
            net = self._residual_block(net, 1 * nf, resample='up', name='res_block4')
            expected_shape(net, [64, 64, 64])
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, **self.bn_params)
            net = slim.conv2d(net, 3, kernel_size=[3, 3], activation_fn=tf.nn.tanh)
            expected_shape(net, [64, 64, 3])
            return net

    def _good_critic(self, X, reuse=False):
        with tf.variable_scope('critic', reuse=reuse):
            nf = 64
            net = slim.conv2d(X, nf, [3, 3], activation_fn=None)
            net = self._residual_block(net, 2 * nf, resample='down', name='res_block1')
            net = self._residual_block(net, 4 * nf, resample='down', name='res_block2')
            net = self._residual_block(net, 8 * nf, resample='down', name='res_block3')
            net = self._residual_block(net, 8 * nf, resample='down', name='res_block4')
            expected_shape(net, [4, 4, 512])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, activation_fn=None)
            return net

class DCGAN(BaseModel):

    def __init__(self, name, training, D_lr=0.0002, G_lr=0.0002, image_shape=[64, 64, 3], z_dim=100):
        self.beta1 = 0.5
        super(DCGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            D_real_prob, D_real_logits = self._discriminator(X)
            D_fake_prob, D_fake_logits = self._discriminator(G, reuse=True)
            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/G/')
            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/G/')
            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars, global_step=global_step)
            self.summary_op = tf.summary.merge([tf.summary.scalar('G_loss', G_loss), tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('D_loss/real', D_loss_real), tf.summary.scalar('D_loss/fake', D_loss_fake)])
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.histogram('real_probs', D_real_prob)
            tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=ops.lrelu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d(net, 64, normalizer_fn=None)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])
            net = slim.flatten(net)
            logits = slim.fully_connected(net, 1, activation_fn=None)
            prob = tf.sigmoid(logits)
            return (prob, logits)

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4 * 4 * 1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])
                return net

class LSGAN(BaseModel):

    def __init__(self, name, training, D_lr=0.001, G_lr=0.001, image_shape=[64, 64, 3], z_dim=1024, a=0.0, b=1.0, c=1.0):
        """
        a: fake label
        b: real label
        c: real label for G (The value that G wants to deceive D - intuitively same as real label b) 

        Pearson chi-square divergence: a=-1, b=1, c=0.
        Intuitive (real label 1, fake label 0): a=0, b=c=1.
        """
        self.a = a
        self.b = b
        self.c = c
        self.beta1 = 0.5
        super(LSGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            D_real = self._discriminator(X)
            D_fake = self._discriminator(G, reuse=True)
            D_loss_real = 0.5 * tf.reduce_mean(tf.square(D_real - self.b))
            D_loss_fake = 0.5 * tf.reduce_mean(tf.square(D_fake - self.a))
            D_loss = D_loss_real + D_loss_fake
            G_loss = 0.5 * tf.reduce_mean(tf.square(D_fake - self.c))
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/G/')
            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/G/')
            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars, global_step=global_step)
            self.summary_op = tf.summary.merge([tf.summary.scalar('G/loss', G_loss), tf.summary.scalar('D/loss', D_loss), tf.summary.scalar('D/loss/real', D_loss_real), tf.summary.scalar('D/loss/fake', D_loss_fake)])
            tf.summary.image('G/fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.histogram('D/real_value', D_real)
            tf.summary.histogram('D/fake_value', D_fake)
            self.all_summary_op = tf.summary.merge_all()
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=ops.lrelu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d(net, 64, normalizer_fn=None)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])
            net = slim.flatten(net)
            d_value = slim.fully_connected(net, 1, activation_fn=None)
            return d_value

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4 * 4 * 256, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params)
            net = tf.reshape(net, [-1, 4, 4, 256])
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[3, 3], padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 256, stride=2)
                net = slim.conv2d_transpose(net, 256, stride=1)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d_transpose(net, 256, stride=2)
                net = slim.conv2d_transpose(net, 256, stride=1)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128, stride=2)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 64, stride=2)
                expected_shape(net, [64, 64, 64])
                net = slim.conv2d_transpose(net, 3, stride=1, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])
                return net

class CoulombGAN(BaseModel):

    def __init__(self, name, training, D_lr=0.00025, G_lr=0.0005, image_shape=[64, 64, 3], z_dim=32):
        self.beta1 = 0.5
        self.kernel_dim = 3
        self.kernel_eps = 1.0
        super(CoulombGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            D_real = self._discriminator(X)
            D_fake = self._discriminator(G, reuse=True)
            '\n            D estimates potential and G minimize D_fake (estimated potential of fake). \n            It means that minimize distance the between real and fake \n            while maximizing the distance between fake and fake.\n\n            P(a) = k(a, real) - k(a, fake).\n            So, \n            P(real) = k(real, real) - k(real, fake),\n            P(fake) = k(fake, real) - k(fake, fake).\n            '
            P_real = calc_potential(G, X, X, kernel_dim=self.kernel_dim, kernel_eps=self.kernel_eps, name='P_real')
            P_fake = calc_potential(G, X, G, kernel_dim=self.kernel_dim, kernel_eps=self.kernel_eps, name='P_fake')
            D_loss_real = tf.losses.mean_squared_error(D_real, P_real)
            D_loss_fake = tf.losses.mean_squared_error(D_fake, P_fake)
            D_loss = D_loss_real + D_loss_fake
            G_loss = -tf.reduce_mean(D_fake)
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/G/')
            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/G/')
            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars, global_step=global_step)
            self.summary_op = tf.summary.merge([tf.summary.scalar('G_loss', G_loss), tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('potential/real_mean', tf.reduce_mean(P_real)), tf.summary.scalar('potential/fake_mean', tf.reduce_mean(P_fake))])
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.histogram('potential/real', P_real)
            tf.summary.histogram('potential/fake', P_fake)
            self.all_summary_op = tf.summary.merge_all()
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=ops.lrelu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d(net, 128, normalizer_fn=None)
                net = slim.conv2d(net, 256)
                net = slim.conv2d(net, 512)
                net = slim.conv2d(net, 1024)
                expected_shape(net, [4, 4, 1024])
            net = slim.flatten(net)
            logits = slim.fully_connected(net, 1, activation_fn=None)
            return logits

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4 * 4 * 1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])
                return net

class EBGAN(BaseModel):

    def __init__(self, name, training, D_lr=0.001, G_lr=0.001, image_shape=[64, 64, 3], z_dim=100, pt_weight=0.1, margin=20.0):
        """ The default value of pt_weight and margin is taken from the paper for celebA. """
        self.pt_weight = pt_weight
        self.m = margin
        self.beta1 = 0.5
        super(EBGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            D_real_latent, D_real_energy = self._discriminator(X)
            D_fake_latent, D_fake_energy = self._discriminator(G, reuse=True)
            D_fake_hinge = tf.maximum(0.0, self.m - D_fake_energy)
            D_loss = D_real_energy + D_fake_hinge
            G_loss = D_fake_energy
            PT = self.pt_regularizer(D_fake_latent)
            pt_loss = self.pt_weight * PT
            G_loss += pt_loss
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/G/')
            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/G/')
            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).minimize(G_loss, var_list=G_vars, global_step=global_step)
            self.summary_op = tf.summary.merge([tf.summary.scalar('G_loss', G_loss), tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('PT', PT), tf.summary.scalar('pt_loss', pt_loss), tf.summary.scalar('D_energy/real', D_real_energy), tf.summary.scalar('D_energy/fake', D_fake_energy), tf.summary.scalar('D_fake_hinge', D_fake_hinge)])
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            self.all_summary_op = tf.summary.merge_all()
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], kernel_size=[4, 4], stride=2, padding='SAME', activation_fn=ops.lrelu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d(net, 64, normalizer_fn=None)
                net = slim.conv2d(net, 128)
                net = slim.conv2d(net, 256)
                latent = net
                expected_shape(latent, [8, 8, 256])
                net = slim.conv2d_transpose(net, 128)
                net = slim.conv2d_transpose(net, 64)
                x_recon = slim.conv2d_transpose(net, 3, activation_fn=None, normalizer_fn=None)
                expected_shape(x_recon, [64, 64, 3])
            energy = tf.sqrt(tf.reduce_sum(tf.square(X - x_recon), axis=[1, 2, 3]))
            energy = tf.reduce_mean(energy)
            return (latent, energy)

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4 * 4 * 1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[4, 4], stride=2, padding='SAME', activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d_transpose(net, 512)
                expected_shape(net, [8, 8, 512])
                net = slim.conv2d_transpose(net, 256)
                expected_shape(net, [16, 16, 256])
                net = slim.conv2d_transpose(net, 128)
                expected_shape(net, [32, 32, 128])
                net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [64, 64, 3])
                return net

    def pt_regularizer(self, lf):
        eps = 1e-08
        lf = slim.flatten(lf)
        l2_norm = tf.norm(lf, axis=1, keep_dims=True)
        expected_shape(l2_norm, [1])
        unit_lf = lf / (l2_norm + eps)
        cos_sim = tf.square(tf.matmul(unit_lf, unit_lf, transpose_b=True))
        N = tf.cast(tf.shape(lf)[0], tf.float32)
        pt_loss = (tf.reduce_sum(cos_sim) - N) / (N * (N - 1))
        return pt_loss


# Cluster 0

def model_fn(features, labels, mode, params):
    indicator_columns, embedding_columns = build_columns()
    fnn = FNN(indicator_columns, embedding_columns, params['warm_up_from_fm'], [64, 32])
    outputs = fnn(features)
    predictions = {'predictions': outputs}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.log_loss(labels, outputs)
    metrics = {'auc': tf.metrics.auc(labels, outputs)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def build_columns():
    movielens = MovielensRanking()
    user_id = tf.feature_column.categorical_column_with_hash_bucket('user_id', movielens.num_users)
    user_gender = tf.feature_column.categorical_column_with_vocabulary_list('user_gender', movielens.gender_vocab)
    user_age = tf.feature_column.categorical_column_with_vocabulary_list('user_age', movielens.age_vocab)
    user_occupation = tf.feature_column.categorical_column_with_vocabulary_list('user_occupation', movielens.occupation_vocab)
    movie_id = tf.feature_column.categorical_column_with_hash_bucket('movie_id', movielens.num_movies)
    movie_genres = tf.feature_column.categorical_column_with_vocabulary_list('movie_genres', movielens.gender_vocab)
    base_columns = [user_id, user_gender, user_age, user_occupation, movie_id, movie_genres]
    indicator_columns = [tf.feature_column.indicator_column(c) for c in base_columns]
    embedding_columns = [tf.feature_column.embedding_column(c, dimension=16) for c in base_columns]
    return (indicator_columns, embedding_columns)

def model_fn(features, labels, mode):
    indicator_columns, embedding_columns = build_columns()
    outputs = FM(indicator_columns, embedding_columns)(features)
    predictions = {'predictions': outputs}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.sigmoid_cross_entropy(labels, outputs)
    metrics = {'auc': tf.metrics.auc(labels, tf.nn.sigmoid(outputs))}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def export_saved_model(estimator, export_path):
    indicator_columns, embedding_columns = build_columns()
    columns = indicator_columns + embedding_columns
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_saved_model(export_path, example_input_fn)

def main():
    movielens = MovielensRanking()
    indicator_columns, embedding_columns = build_columns()
    model = DeepFM(indicator_columns, embedding_columns, dnn_units_size=[256, 32])
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.fit(movielens.training_input_fn, epochs=10, steps_per_epoch=movielens.train_steps_per_epoch, validation_data=movielens.testing_input_fn, validation_steps=movielens.test_steps, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

def model_fn(features, labels, mode):
    indicator_columns, embedding_columns = build_columns()
    outputs = DeepFM(indicator_columns, embedding_columns, [64, 32])(features)
    predictions = {'predictions': outputs}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.log_loss(labels, outputs)
    metrics = {'auc': tf.metrics.auc(labels, outputs)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def model_fn(features, labels, mode):
    columns = build_columns()
    outputs = MMoE(columns, num_tasks=2, num_experts=2, task_hidden_units=[32, 10], expert_hidden_units=[64, 32])(features)
    predictions = {'predictions0': outputs[0], 'predictions1': outputs[1]}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    labels0 = tf.expand_dims(labels['labels0'], axis=1)
    labels1 = tf.expand_dims(labels['labels1'], axis=1)
    loss0 = tf.losses.mean_squared_error(labels=labels0, predictions=outputs[0])
    loss1 = tf.losses.mean_squared_error(labels=labels1, predictions=outputs[1])
    total_loss = loss0 + loss1
    tf.summary.scalar('task0_loss', loss0)
    tf.summary.scalar('task1_loss', loss1)
    tf.summary.scalar('total_loss', total_loss)
    metrics = {'task0_mse': tf.metrics.mean_squared_error(labels0, outputs[0]), 'task1_mse': tf.metrics.mean_squared_error(labels1, outputs[1])}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = tf.group(optimizer.minimize(loss=loss0, global_step=tf.train.get_global_step()), optimizer.minimize(loss=loss1, global_step=tf.train.get_global_step()))
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

class TestFM(tf.test.TestCase):

    def test_fm_layer(self):
        sparse_inputs = np.random.randint(0, 2, size=(10, 10)).astype(np.float32)
        embedding_inputs = np.random.normal(size=(10, 5, 5)).astype(np.float32)
        x_sum = np.sum(embedding_inputs, axis=1)
        x_square_sum = np.sum(np.power(embedding_inputs, 2), axis=1)
        expected_outputs = 0.5 * np.sum(np.power(x_sum, 2) - x_square_sum, axis=1, keepdims=True)
        outputs = FM()(sparse_inputs, embedding_inputs)
        self.assertAllClose(outputs, expected_outputs)

    def test_fm_layer_train(self):

        def get_model():
            sparse_inputs = tf.keras.layers.Input(shape=(10,))
            embedding_inputs = tf.keras.layers.Input(shape=(5, 5))
            x = FM()(sparse_inputs, embedding_inputs)
            logits = tf.keras.layers.Dense(1)(x)
            return tf.keras.Model([sparse_inputs, embedding_inputs], logits)
        model = get_model()
        random_sparse_inputs = np.random.randint(0, 2, size=(10, 10))
        random_embedding_inputs = np.random.uniform(size=(10, 5, 5))
        random_outputs = np.random.uniform(size=(10,))
        model.compile(loss='mse')
        model.fit([random_sparse_inputs, random_embedding_inputs], random_outputs, verbose=0)

    def test_fm_layer_save(self):

        def get_model():
            sparse_inputs = tf.keras.layers.Input(shape=(10,))
            embedding_inputs = tf.keras.layers.Input(shape=(5, 5))
            x = FM()(sparse_inputs, embedding_inputs)
            logits = tf.keras.layers.Dense(1)(x)
            return tf.keras.Model([sparse_inputs, embedding_inputs], logits)
        model = get_model()
        random_sparse_inputs = np.random.randint(0, 2, size=(10, 10))
        random_embedding_inputs = np.random.uniform(size=(10, 5, 5))
        model_pred = model.predict([random_sparse_inputs, random_embedding_inputs])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'fm')
            model.save(path, options=tf.saved_model.SaveOptions(namespace_whitelist=['FM']))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict([random_sparse_inputs, random_embedding_inputs])
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)

    def test_model(self):

        def build_columns():
            user_id = tf.feature_column.categorical_column_with_hash_bucket('user_id', 100)
            movie_id = tf.feature_column.categorical_column_with_hash_bucket('movie_id', 100)
            base_columns = [user_id, movie_id]
            _indicator_columns = [tf.feature_column.indicator_column(c) for c in base_columns]
            _embedding_columns = [tf.feature_column.embedding_column(c, dimension=16) for c in base_columns]
            return (_indicator_columns, _embedding_columns)
        indicator_columns, embedding_columns = build_columns()
        model = FactorizationMachine(indicator_columns, embedding_columns)
        model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam())
        dataset = tf.data.Dataset.from_tensor_slices(({'user_id': [['1']] * 1000, 'movie_id': [['2']] * 1000}, np.random.randint(0, 1, size=(1000, 1))))
        model.fit(dataset, steps_per_epoch=100, verbose=-1)
        test_data = {'user_id': np.asarray([['1'], ['2']]), 'movie_id': np.asarray([['1'], ['2']])}
        model_pred = model.predict(test_data)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'FM')
            model.save(path)
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(test_data)
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)

class TestDeepFM(tf.test.TestCase):

    def test_model_train(self):

        def build_columns():
            user_id = tf.feature_column.categorical_column_with_hash_bucket('user_id', 100)
            movie_id = tf.feature_column.categorical_column_with_hash_bucket('movie_id', 100)
            base_columns = [user_id, movie_id]
            _indicator_columns = [tf.feature_column.indicator_column(c) for c in base_columns]
            _embedding_columns = [tf.feature_column.embedding_column(c, dimension=16) for c in base_columns]
            return (_indicator_columns, _embedding_columns)
        indicator_columns, embedding_columns = build_columns()
        model = DeepFM(indicator_columns, embedding_columns, dnn_units_size=[10, 5])
        model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam())
        dataset = tf.data.Dataset.from_tensor_slices(({'user_id': [['1']] * 1000, 'movie_id': [['2']] * 1000}, np.random.randint(0, 1, size=(1000, 1))))
        model.fit(dataset, steps_per_epoch=100, verbose=-1)
        test_data = {'user_id': np.asarray([['1'], ['2']]), 'movie_id': np.asarray([['1'], ['2']])}
        model_pred = model.predict(test_data)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'FM')
            model.save(path)
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict(test_data)
        for model_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            assert model_layer.get_config() == loaded_layer.get_config()
        self.assertAllEqual(model_pred, loaded_pred)

class TestMixtureOfExperts(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(32, 64, 128, 512)
    def test_mmoe(self, batch_size):

        def build_columns():
            return [tf.feature_column.numeric_column('C{}'.format(i)) for i in range(100)]
        columns = build_columns()
        model = MMoE(columns, num_tasks=2, num_experts=2, task_hidden_units=[32, 10], expert_hidden_units=[64, 32])
        dataset = SyntheticForMultiTask(5000)
        with self.session() as sess:
            iterator = tf.data.make_one_shot_iterator(dataset.input_fn(batch_size=batch_size))
            x, y = iterator.get_next()
            y_pred = model(x)
            sess.run(tf.global_variables_initializer())
            a = sess.run(y_pred[0])
            b = sess.run(y_pred[1])
            self.assertAllEqual(len(y_pred), 2)
            self.assertAllEqual(a.shape, (batch_size, 1))
            self.assertAllEqual(b.shape, (batch_size, 1))

class TestESMM(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(32, 64, 128, 512)
    def test_mmoe(self, batch_size):

        def build_columns():
            return [tf.feature_column.numeric_column('C{}'.format(i)) for i in range(100)]
        columns = build_columns()
        model = ESMM(columns, hidden_units=[32, 10])
        dataset = SyntheticForMultiTask(5000)
        with self.session() as sess:
            iterator = tf.data.make_one_shot_iterator(dataset.input_fn(batch_size=batch_size))
            x, y = iterator.get_next()
            p_cvr, p_ctr, p_ctcvr = model(x)
            sess.run(tf.global_variables_initializer())
            p_cvr = sess.run(p_cvr)
            p_ctr = sess.run(p_ctr)
            p_ctcvr = sess.run(p_ctcvr)
            self.assertAllEqual(p_cvr.shape, (batch_size, 1))
            self.assertAllEqual(p_ctr.shape, (batch_size, 1))
            self.assertAllEqual(p_ctcvr.shape, (batch_size, 1))


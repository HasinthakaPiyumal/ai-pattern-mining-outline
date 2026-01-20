# Cluster 6

def train_model():
    cora = Cora()
    ids, features, labels = cora.load_content()
    graph = cora.build_graph(ids)
    spectral_graph = cora.spectral_graph(graph)
    cora.sample_train_nodes(labels)
    train, valid, test = cora.split_labels(labels)

    def build_model():
        g = tf.keras.layers.Input(shape=(None,))
        feats = tf.keras.layers.Input(shape=(features.shape[-1],))
        x = GCN(32)(feats, g)
        outputs = GCN(cora.num_classes, activation='softmax')(x, g)
        return tf.keras.Model([g, feats], outputs)
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='categorical_crossentropy', weighted_metrics=['acc'])
    train_labels, train_mask = train
    valid_labels, valid_mask = valid
    test_labels, test_mask = test
    batch_size = graph.shape[0]
    model.fit([spectral_graph, features], train_labels, sample_weight=train_mask, validation_data=([spectral_graph, features], valid_labels, valid_mask), batch_size=batch_size, epochs=200, shuffle=False, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
    eval_results = model.evaluate([spectral_graph, features], test_labels, sample_weight=test_mask, batch_size=batch_size, verbose=0)
    print('Test Loss: {:.4f}'.format(eval_results[0]))
    print('Test Accuracy: {:.4f}'.format(eval_results[1]))

def build_model():
    g = tf.keras.layers.Input(shape=(None,))
    feats = tf.keras.layers.Input(shape=(features.shape[-1],))
    x = GCN(32)(feats, g)
    outputs = GCN(cora.num_classes, activation='softmax')(x, g)
    return tf.keras.Model([g, feats], outputs)

def train_model(vocab_size=5000, max_len=128, batch_size=128, epochs=10):
    train, test = load_dataset(vocab_size, max_len)
    x_train, x_train_masks, y_train = train
    x_test, x_test_masks, y_test = test
    model = build_model(vocab_size, max_len)
    model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-09), loss='categorical_crossentropy', metrics=['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(patience=3)
    model.fit([x_train, x_train_masks], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[es])
    test_metrics = model.evaluate([x_test, x_test_masks], y_test, batch_size=batch_size, verbose=0)
    print('loss on Test: %.4f' % test_metrics[0])
    print('accu on Test: %.4f' % test_metrics[1])

def load_dataset(vocab_size, max_len):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(maxlen=max_len, num_words=vocab_size)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    x_train_masks = tf.equal(x_train, 0)
    x_test_masks = tf.equal(x_test, 0)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return ((x_train, x_train_masks, y_train), (x_test, x_test_masks, y_test))

class TestDIN(tf.test.TestCase, parameterized.TestCase):

    def test_activation_unit_noiteract(self):
        x = np.random.normal(size=(3, 5))
        y = np.random.normal(size=(3, 5))
        activation_unit = din.ActivationUnit(10, kernel_init='ones')
        outputs = activation_unit(x, y)
        dense = tf.keras.layers.Dense(10, activation='relu', kernel_initializer='ones')
        expected_outputs = tf.math.reduce_sum(dense(np.concatenate([x, y], axis=1)), axis=1, keepdims=True)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expected_outputs)

    def test_activation_unit_iteract(self):
        x = np.random.normal(size=(3, 5))
        y = np.random.normal(size=(3, 5))
        interacter = tf.keras.layers.Subtract()
        activation_unit = din.ActivationUnit(10, interacter=interacter, kernel_init='ones')
        outputs = activation_unit(x, y)
        dense = tf.keras.layers.Dense(10, activation='relu', kernel_initializer='ones')
        expected_outputs = tf.math.reduce_sum(dense(np.concatenate([x, y, x - y], axis=1)), axis=1, keepdims=True)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expected_outputs)

    @parameterized.parameters(1e-07, 1e-08, 1e-09, 1e-10)
    def test_dice(self, epsilon):
        inputs = np.asarray([[-0.2, -0.1, 0.1, 0.2]]).astype(np.float32)
        outputs = din.Dice(epsilon=epsilon)(inputs)
        p = (inputs - inputs.mean()) / np.math.sqrt(inputs.std() + epsilon)
        p = 1 / (1 + np.exp(-p))
        x = tf.where(inputs > 0, x=inputs, y=tf.zeros_like(inputs))
        expected_outputs = tf.where(x > 0, x=p * x, y=(1 - p) * x)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(outputs, expected_outputs)

    def test_din(self):

        def build_model():
            x = tf.keras.layers.Input(shape=(5,))
            y = tf.keras.layers.Input(shape=(5,))
            interacter = tf.keras.layers.Subtract()
            activation_unit = din.ActivationUnit(10, interacter=interacter)
            outputs = activation_unit(x, y)
            return tf.keras.Model([x, y], outputs)
        x_embeddings = np.random.normal(size=(10, 5))
        y_embeddings = np.random.normal(size=(10, 5))
        labels = np.random.normal(size=(10,))
        model = build_model()
        model.compile(loss='mse')
        model.fit([x_embeddings, y_embeddings], labels, verbose=0)
        model_pred = model.predict([x_embeddings, y_embeddings])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'din_model')
            model.save(path, options=tf.saved_model.SaveOptions(namespace_whitelist=['din']))
            loaded_model = tf.keras.models.load_model(path)
            loaded_pred = loaded_model.predict([x_embeddings, y_embeddings])
        self.assertAllEqual(model_pred, loaded_pred)


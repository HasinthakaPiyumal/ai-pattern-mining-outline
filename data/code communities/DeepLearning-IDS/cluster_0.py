# Cluster 0

def experiment(dataFile, optimizer, epochs, batch_size):
    seed = 7
    numpy.random.seed(seed)
    cvscores = []
    print('optimizer: {} epochs: {} batch_size: {}'.format(optimizer, epochs, batch_size))
    data = loadData(dataFile)
    data_y = data.pop('Label').values
    data_x = data.as_matrix()
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in kfold.split(data_x, data_y):
        model = Sequential()
        model.add(Dense(80, activation='relu', input_dim=80))
        model.add(Dense(80, activation='relu', input_dim=80))
        model.add(Dense(1, activation='softmax'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(data_x[train], data_y[train], epochs=10, batch_size=100, verbose=0)
        scores = model.evaluate(data_x[test], data_y[test], verbose=0)
        print('{}: {:.2f}%'.format(model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        resultFile = os.path.join(resultPath, dataFile)
        with open('{}.result'.format(resultFile), 'a') as fout:
            fout.write('accuracy: {:.2f} std-dev: {:.2f}\n'.format(np.mean(cvscores), np.std(cvscores)))


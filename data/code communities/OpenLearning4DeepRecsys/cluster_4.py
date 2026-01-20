# Cluster 4

def run():
    print('begin running')
    field_cnt = 46
    feature_cnt = 46
    params = {'reg_w_linear': 0.0001, 'reg_w_fm': 0.0001, 'reg_w_nn': 0.0001, 'reg_w_l1': 0.0001, 'init_value': 0.1, 'layer_sizes': [10, 5], 'keep_probs': [0.7, 0.7], 'activations': ['tanh', 'tanh'], 'eta': 0.1, 'n_epoch': 5000, 'batch_size': 50, 'dim': 8, 'model_path': 'models', 'log_path': 'logs/' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S'), 'train_file': 'data/S1_4.txt', 'test_file': 'data/S5.txt', 'output_predictions': False, 'is_use_fm_part': True, 'is_use_dnn_part': True, 'learning_rate': 0.01, 'loss': 'log_loss', 'optimizer': 'sgd'}
    single_run(feature_cnt, field_cnt, params)


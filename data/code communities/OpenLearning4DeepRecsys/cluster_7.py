# Cluster 7

def grid_search_params():
    dataset = data_reader.sparse_data_repos(10000, 10005)
    dataset.load_trainging_ratings('data/userbook_unique_compactid_train.txt')
    dataset.load_test_ratings('data/userbook_unique_compactid_valid.txt')
    dataset.load_eval_ratings('data/userbook_unique_compactid_test.txt')
    log_file = 'logs/BMF_book.csv'
    wt = open(log_file, 'w')
    rank = 16
    lambs = [3e-05, 5e-05, 0.0001]
    batch_sizes = [500]
    n_eopch = 2000
    lrs = [0.1]
    init_values = [0.01]
    mu = np.asarray(dataset.training_ratings_score, dtype=np.float32).mean()
    wt.write('rank,lr,lamb,mu,n_eopch,batch_size,best_train_rmse,best_test_rmse,best_eval_rmse,best_epoch,init_value,minutes\n')
    for lamb in lambs:
        for lr in lrs:
            for init_value in init_values:
                for batch_size in batch_sizes:
                    run_with_parameter(dataset, rank, lr, lamb, mu, n_eopch, batch_size, wt, init_value)
    wt.close()


# Cluster 0

def evaluate_model(sess, model, dataset, topk):
    hits, ndcgs = ([], [])
    for u, i in dataset.testPosSet:
        hit, ndcg = evaluate_one_case(u, i, dataset.testPair2NegList, sess, model, topk)
        hits.append(hit)
        ndcgs.append(ndcg)
    return (np.asarray(hits).mean(), np.asarray(ndcgs).mean())

def evaluate_one_case(u, i, key2candidates, sess, model, topk):
    key = (u, i)
    assert key in key2candidates
    items = key2candidates[key]
    users = np.full(len(items), key[0], dtype=np.int32)
    predictions = sess.run(model.output, {model.user_indices: users, model.item_indices: items})
    k = min(topk, len(items))
    sorted_idx = np.argsort(predictions)[::-1]
    selected_items = items[sorted_idx[0:k]]
    ndcg = getNDCG(selected_items, i)
    hit = getHitRatio(selected_items, i)
    return (hit, ndcg)

def single_run(args, dataset):
    model = NeuMF(args, dataset.num_users, dataset.num_items)
    model.build_model()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    t1 = time()
    ahit, andcg = evaluate_model(sess, model, dataset, args.topk)
    best_hr, best_ndcg, best_iter = (ahit, andcg, -1)
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (ahit, andcg, time() - t1))
    for epoch in range(args.epochs):
        t1 = time()
        train_users, train_items, train_labels, num_inst = dataset.make_training_instances(args.num_neg_inst)
        loss_per_epoch, error_per_epoch = (0, 0)
        for ite in range((num_inst - 1) // args.batch_size + 1):
            start_idx = ite * args.batch_size
            end_idx = min((ite + 1) * args.batch_size, num_inst)
            cur_user_indices, cur_item_indices, cur_label = (train_users[start_idx:end_idx], train_items[start_idx:end_idx], train_labels[start_idx:end_idx])
            _, loss, error = sess.run([model.train_step, model.loss, model.raw_error], {model.user_indices: cur_user_indices, model.item_indices: cur_item_indices, model.ratings: cur_label})
            loss_per_epoch += loss
            error_per_epoch += error
        error_per_epoch /= num_inst
        t2 = time()
        if epoch % args.verbose == 0:
            ahit, andcg = evaluate_model(sess, model, dataset, args.topk)
            print('epoch %d   \t[%.1f s]: HR= %.4f\tNDCG= %.4f\tloss= %.4f\terror= %.4f\t[%.1f s]' % (epoch, t2 - t1, ahit, andcg, loss_per_epoch, error_per_epoch, time() - t2))
            if ahit > best_hr:
                best_hr = ahit
                best_iter = epoch
            if andcg > best_ndcg:
                best_ndcg = andcg
    print('End. Best Epoch %d:  HR = %.4f, NDCG = %.4f. ' % (best_iter, best_hr, best_ndcg))


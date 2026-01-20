# Cluster 64

def save_global_samples(global_mask_fnames, mask2real_fname, mask2fake_fname, out_dir, real_scores_by_fname, fake_scores_by_fname):
    for cur_mask_fname in global_mask_fnames:
        cur_real_fname = mask2real_fname[cur_mask_fname]
        orig_img = load_image(cur_real_fname, mode='RGB')
        fake_img = load_image(mask2fake_fname[cur_mask_fname], mode='RGB')[:, :orig_img.shape[1], :orig_img.shape[2]]
        mask = load_image(cur_mask_fname, mode='L')[None, ...]
        draw_score(orig_img, real_scores_by_fname.loc[cur_real_fname, 'real_score'])
        draw_score(fake_img, fake_scores_by_fname.loc[cur_mask_fname, 'fake_score'])
        cur_grid = visualize_mask_and_images(dict(image=orig_img, mask=mask, fake=fake_img), keys=['image', 'fake'], last_without_mask=True)
        cur_grid = np.clip(cur_grid * 255, 0, 255).astype('uint8')
        cur_grid = cv2.cvtColor(cur_grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(cur_mask_fname))[0] + '.jpg'), cur_grid)

def draw_score(img, score):
    img = np.transpose(img, (1, 2, 0))
    cv2.putText(img, f'{score:.2f}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 1, 0), thickness=3)
    img = np.transpose(img, (2, 0, 1))
    return img

def save_samples_by_real(worst_best_by_real, mask2fake_fname, fake_info, out_dir):
    for real_fname in worst_best_by_real.index:
        worst_mask_path = worst_best_by_real.loc[real_fname, 'worst']
        best_mask_path = worst_best_by_real.loc[real_fname, 'best']
        orig_img = load_image(real_fname, mode='RGB')
        worst_mask_img = load_image(worst_mask_path, mode='L')[None, ...]
        worst_fake_img = load_image(mask2fake_fname[worst_mask_path], mode='RGB')[:, :orig_img.shape[1], :orig_img.shape[2]]
        best_mask_img = load_image(best_mask_path, mode='L')[None, ...]
        best_fake_img = load_image(mask2fake_fname[best_mask_path], mode='RGB')[:, :orig_img.shape[1], :orig_img.shape[2]]
        draw_score(orig_img, worst_best_by_real.loc[real_fname, 'real_score'])
        draw_score(worst_fake_img, worst_best_by_real.loc[real_fname, 'worst_score'])
        draw_score(best_fake_img, worst_best_by_real.loc[real_fname, 'best_score'])
        cur_grid = visualize_mask_and_images(dict(image=orig_img, mask=np.zeros_like(worst_mask_img), worst_mask=worst_mask_img, worst_img=worst_fake_img, best_mask=best_mask_img, best_img=best_fake_img), keys=['image', 'worst_mask', 'worst_img', 'best_mask', 'best_img'], rescale_keys=['worst_mask', 'best_mask'], last_without_mask=True)
        cur_grid = np.clip(cur_grid * 255, 0, 255).astype('uint8')
        cur_grid = cv2.cvtColor(cur_grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(real_fname))[0] + '.jpg'), cur_grid)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        cur_stat = fake_info[fake_info['real_fname'] == real_fname]
        cur_stat['fake_score'].hist(ax=ax1)
        cur_stat['real_score'].hist(ax=ax2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, os.path.splitext(os.path.basename(real_fname))[0] + '_scores.png'))
        plt.close(fig)

def main(args):
    config = load_yaml(args.config)
    latents_dir = os.path.join(args.outpath, 'latents')
    os.makedirs(latents_dir, exist_ok=True)
    global_worst_dir = os.path.join(args.outpath, 'global_worst')
    os.makedirs(global_worst_dir, exist_ok=True)
    global_best_dir = os.path.join(args.outpath, 'global_best')
    os.makedirs(global_best_dir, exist_ok=True)
    worst_best_by_best_worst_score_diff_max_dir = os.path.join(args.outpath, 'worst_best_by_real', 'best_worst_score_diff_max')
    os.makedirs(worst_best_by_best_worst_score_diff_max_dir, exist_ok=True)
    worst_best_by_best_worst_score_diff_min_dir = os.path.join(args.outpath, 'worst_best_by_real', 'best_worst_score_diff_min')
    os.makedirs(worst_best_by_best_worst_score_diff_min_dir, exist_ok=True)
    worst_best_by_real_best_score_diff_max_dir = os.path.join(args.outpath, 'worst_best_by_real', 'real_best_score_diff_max')
    os.makedirs(worst_best_by_real_best_score_diff_max_dir, exist_ok=True)
    worst_best_by_real_best_score_diff_min_dir = os.path.join(args.outpath, 'worst_best_by_real', 'real_best_score_diff_min')
    os.makedirs(worst_best_by_real_best_score_diff_min_dir, exist_ok=True)
    worst_best_by_real_worst_score_diff_max_dir = os.path.join(args.outpath, 'worst_best_by_real', 'real_worst_score_diff_max')
    os.makedirs(worst_best_by_real_worst_score_diff_max_dir, exist_ok=True)
    worst_best_by_real_worst_score_diff_min_dir = os.path.join(args.outpath, 'worst_best_by_real', 'real_worst_score_diff_min')
    os.makedirs(worst_best_by_real_worst_score_diff_min_dir, exist_ok=True)
    if not args.only_report:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception_model = InceptionV3([block_idx]).eval().cuda()
        dataset = PrecomputedInpaintingResultsDataset(args.datadir, args.predictdir, **config.dataset_kwargs)
        real2vector_cache = {}
        real_features = []
        fake_features = []
        orig_fnames = []
        mask_fnames = []
        mask2real_fname = {}
        mask2fake_fname = {}
        for batch_i, batch in enumerate(dataset):
            orig_img_fname = dataset.img_filenames[batch_i]
            mask_fname = dataset.mask_filenames[batch_i]
            fake_fname = dataset.pred_filenames[batch_i]
            mask2real_fname[mask_fname] = orig_img_fname
            mask2fake_fname[mask_fname] = fake_fname
            cur_real_vector = real2vector_cache.get(orig_img_fname, None)
            if cur_real_vector is None:
                with torch.no_grad():
                    in_img = torch.from_numpy(batch['image'][None, ...]).cuda()
                    cur_real_vector = inception_model(in_img)[0].squeeze(-1).squeeze(-1).cpu().numpy()
                real2vector_cache[orig_img_fname] = cur_real_vector
            pred_img = torch.from_numpy(batch['inpainted'][None, ...]).cuda()
            cur_fake_vector = inception_model(pred_img)[0].squeeze(-1).squeeze(-1).cpu().numpy()
            real_features.append(cur_real_vector)
            fake_features.append(cur_fake_vector)
            orig_fnames.append(orig_img_fname)
            mask_fnames.append(mask_fname)
        ids_features = np.concatenate(real_features + fake_features, axis=0)
        ids_labels = np.array([1] * len(real_features) + [0] * len(fake_features))
        with open(os.path.join(latents_dir, 'featues.pkl'), 'wb') as f:
            pickle.dump(ids_features, f, protocol=3)
        with open(os.path.join(latents_dir, 'labels.pkl'), 'wb') as f:
            pickle.dump(ids_labels, f, protocol=3)
        with open(os.path.join(latents_dir, 'orig_fnames.pkl'), 'wb') as f:
            pickle.dump(orig_fnames, f, protocol=3)
        with open(os.path.join(latents_dir, 'mask_fnames.pkl'), 'wb') as f:
            pickle.dump(mask_fnames, f, protocol=3)
        with open(os.path.join(latents_dir, 'mask2real_fname.pkl'), 'wb') as f:
            pickle.dump(mask2real_fname, f, protocol=3)
        with open(os.path.join(latents_dir, 'mask2fake_fname.pkl'), 'wb') as f:
            pickle.dump(mask2fake_fname, f, protocol=3)
        svm = sklearn.svm.LinearSVC(dual=False)
        svm.fit(ids_features, ids_labels)
        pred_scores = svm.decision_function(ids_features)
        real_scores = pred_scores[:len(real_features)]
        fake_scores = pred_scores[len(real_features):]
        with open(os.path.join(latents_dir, 'pred_scores.pkl'), 'wb') as f:
            pickle.dump(pred_scores, f, protocol=3)
        with open(os.path.join(latents_dir, 'real_scores.pkl'), 'wb') as f:
            pickle.dump(real_scores, f, protocol=3)
        with open(os.path.join(latents_dir, 'fake_scores.pkl'), 'wb') as f:
            pickle.dump(fake_scores, f, protocol=3)
    else:
        with open(os.path.join(latents_dir, 'orig_fnames.pkl'), 'rb') as f:
            orig_fnames = pickle.load(f)
        with open(os.path.join(latents_dir, 'mask_fnames.pkl'), 'rb') as f:
            mask_fnames = pickle.load(f)
        with open(os.path.join(latents_dir, 'mask2real_fname.pkl'), 'rb') as f:
            mask2real_fname = pickle.load(f)
        with open(os.path.join(latents_dir, 'mask2fake_fname.pkl'), 'rb') as f:
            mask2fake_fname = pickle.load(f)
        with open(os.path.join(latents_dir, 'real_scores.pkl'), 'rb') as f:
            real_scores = pickle.load(f)
        with open(os.path.join(latents_dir, 'fake_scores.pkl'), 'rb') as f:
            fake_scores = pickle.load(f)
    real_info = pd.DataFrame(data=[dict(real_fname=fname, real_score=score) for fname, score in zip(orig_fnames, real_scores)])
    real_info.set_index('real_fname', drop=True, inplace=True)
    fake_info = pd.DataFrame(data=[dict(mask_fname=fname, fake_fname=mask2fake_fname[fname], real_fname=mask2real_fname[fname], fake_score=score) for fname, score in zip(mask_fnames, fake_scores)])
    fake_info = fake_info.join(real_info, on='real_fname', how='left')
    fake_info.drop_duplicates(['fake_fname', 'real_fname'], inplace=True)
    fake_stats_by_real = fake_info.groupby('real_fname')['fake_score'].describe()[['mean', 'std']].rename({'mean': 'mean_fake_by_real', 'std': 'std_fake_by_real'}, axis=1)
    fake_info = fake_info.join(fake_stats_by_real, on='real_fname', rsuffix='stat_by_real')
    fake_info.drop_duplicates(['fake_fname', 'real_fname'], inplace=True)
    fake_info.to_csv(os.path.join(latents_dir, 'join_scores_table.csv'), sep='\t', index=False)
    fake_scores_table = fake_info.set_index('mask_fname')['fake_score'].to_frame()
    real_scores_table = fake_info.set_index('real_fname')['real_score'].drop_duplicates().to_frame()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(fake_scores)
    ax2.hist(real_scores)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outpath, 'global_scores_hist.png'))
    plt.close(fig)
    global_worst_masks = fake_info.sort_values('fake_score', ascending=True)['mask_fname'].iloc[:config.take_global_top].to_list()
    global_best_masks = fake_info.sort_values('fake_score', ascending=False)['mask_fname'].iloc[:config.take_global_top].to_list()
    save_global_samples(global_worst_masks, mask2real_fname, mask2fake_fname, global_worst_dir, real_scores_table, fake_scores_table)
    save_global_samples(global_best_masks, mask2real_fname, mask2fake_fname, global_best_dir, real_scores_table, fake_scores_table)
    worst_samples_by_real = fake_info.groupby('real_fname').apply(lambda d: d.set_index('mask_fname')['fake_score'].idxmin()).to_frame().rename({0: 'worst'}, axis=1)
    best_samples_by_real = fake_info.groupby('real_fname').apply(lambda d: d.set_index('mask_fname')['fake_score'].idxmax()).to_frame().rename({0: 'best'}, axis=1)
    worst_best_by_real = pd.concat([worst_samples_by_real, best_samples_by_real], axis=1)
    worst_best_by_real = worst_best_by_real.join(fake_scores_table.rename({'fake_score': 'worst_score'}, axis=1), on='worst')
    worst_best_by_real = worst_best_by_real.join(fake_scores_table.rename({'fake_score': 'best_score'}, axis=1), on='best')
    worst_best_by_real = worst_best_by_real.join(real_scores_table)
    worst_best_by_real['best_worst_score_diff'] = worst_best_by_real['best_score'] - worst_best_by_real['worst_score']
    worst_best_by_real['real_best_score_diff'] = worst_best_by_real['real_score'] - worst_best_by_real['best_score']
    worst_best_by_real['real_worst_score_diff'] = worst_best_by_real['real_score'] - worst_best_by_real['worst_score']
    worst_best_by_best_worst_score_diff_min = worst_best_by_real.sort_values('best_worst_score_diff', ascending=True).iloc[:config.take_worst_best_top]
    worst_best_by_best_worst_score_diff_max = worst_best_by_real.sort_values('best_worst_score_diff', ascending=False).iloc[:config.take_worst_best_top]
    save_samples_by_real(worst_best_by_best_worst_score_diff_min, mask2fake_fname, fake_info, worst_best_by_best_worst_score_diff_min_dir)
    save_samples_by_real(worst_best_by_best_worst_score_diff_max, mask2fake_fname, fake_info, worst_best_by_best_worst_score_diff_max_dir)
    worst_best_by_real_best_score_diff_min = worst_best_by_real.sort_values('real_best_score_diff', ascending=True).iloc[:config.take_worst_best_top]
    worst_best_by_real_best_score_diff_max = worst_best_by_real.sort_values('real_best_score_diff', ascending=False).iloc[:config.take_worst_best_top]
    save_samples_by_real(worst_best_by_real_best_score_diff_min, mask2fake_fname, fake_info, worst_best_by_real_best_score_diff_min_dir)
    save_samples_by_real(worst_best_by_real_best_score_diff_max, mask2fake_fname, fake_info, worst_best_by_real_best_score_diff_max_dir)
    worst_best_by_real_worst_score_diff_min = worst_best_by_real.sort_values('real_worst_score_diff', ascending=True).iloc[:config.take_worst_best_top]
    worst_best_by_real_worst_score_diff_max = worst_best_by_real.sort_values('real_worst_score_diff', ascending=False).iloc[:config.take_worst_best_top]
    save_samples_by_real(worst_best_by_real_worst_score_diff_min, mask2fake_fname, fake_info, worst_best_by_real_worst_score_diff_min_dir)
    save_samples_by_real(worst_best_by_real_worst_score_diff_max, mask2fake_fname, fake_info, worst_best_by_real_worst_score_diff_max_dir)
    overlapping_mask_fname_pairs = []
    overlapping_mask_fname_score_diffs = []
    for cur_real_fname in orig_fnames:
        cur_fakes_info = fake_info[fake_info['real_fname'] == cur_real_fname]
        cur_mask_fnames = sorted(cur_fakes_info['mask_fname'].unique())
        cur_mask_pairs_and_scores = Parallel(args.n_jobs)((delayed(extract_overlapping_masks)(cur_mask_fnames, i, fake_scores_table) for i in range(len(cur_mask_fnames) - 1)))
        for cur_pairs, cur_scores in cur_mask_pairs_and_scores:
            overlapping_mask_fname_pairs.extend(cur_pairs)
            overlapping_mask_fname_score_diffs.extend(cur_scores)
    overlapping_mask_fname_pairs = np.asarray(overlapping_mask_fname_pairs)
    overlapping_mask_fname_score_diffs = np.asarray(overlapping_mask_fname_score_diffs)
    overlapping_sort_idx = np.argsort(overlapping_mask_fname_score_diffs)
    overlapping_mask_fname_pairs = overlapping_mask_fname_pairs[overlapping_sort_idx]
    overlapping_mask_fname_score_diffs = overlapping_mask_fname_score_diffs[overlapping_sort_idx]


# Cluster 11

def mean_ensemble():
    all_test_items = {}
    for sub_file in ensemble_subs:
        sub_file_path = os.path.join(subs_dir, sub_file)
        df = pd.read_csv(sub_file_path, header=0)
        all_predict = df.values.tolist()
        for records in all_predict:
            file_id = records[0]
            preds = records[1:]
            if file_id in all_test_items:
                all_test_items[file_id].append(preds)
            else:
                all_test_items[file_id] = [preds]
    cur_record = 0
    df = pd.DataFrame(columns=['image_id', 'image_category'] + config.all_keys)
    num_keypoints_plus = len(config.all_keys) + 1
    for k, v in all_test_items.items():
        temp_list = []
        len_pred = len(v) * 1.0
        for pred_ind in range(1, num_keypoints_plus):
            pred_x, pred_y, pred_v = (0.0, 0.0, 1)
            if v[0][pred_ind].strip() == '-1_-1_-1':
                temp_list.append('-1_-1_-1')
                continue
            for _pred in v:
                _pred_x, _pred_y, _pred_v = _pred[pred_ind].strip().split('_')
                _pred_x, _pred_y, _pred_v = (float(_pred_x), float(_pred_y), int(_pred_v))
                pred_x = pred_x + _pred_x / len_pred
                pred_y = pred_y + _pred_y / len_pred
            temp_list.append('{}_{}_{}'.format(round(pred_x), round(pred_y), pred_v))
        df.loc[cur_record] = [k, v[0][0]] + temp_list
        cur_record = cur_record + 1
    df.sort_values('image_id').to_csv(os.path.join(subs_dir, 'ensmeble.csv'), encoding='utf-8', index=False)


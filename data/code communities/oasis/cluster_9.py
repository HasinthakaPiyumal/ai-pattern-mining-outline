# Cluster 9

def visualization(up_result, down_result, control_result, exp_name, folder_path):
    up_mean, up_ci_low, up_ci_high = mean_confidence_interval(up_result)
    down_mean, down_ci_low, down_ci_high = mean_confidence_interval(down_result)
    control_mean, control_ci_low, control_ci_high = mean_confidence_interval(control_result)
    labels = ['Down', 'Control', 'Up']
    means = [down_mean, control_mean, up_mean]
    conf_intervals = [(down_ci_low, down_ci_high), (control_ci_low, control_ci_high), (up_ci_low, up_ci_high)]
    x_pos = range(len(labels))
    fig, ax = plt.subplots()
    ax.bar(labels, means, color='skyblue', yerr=np.transpose([[mean - ci_low, ci_high - mean] for mean, (ci_low, ci_high) in zip(means, conf_intervals)]), capsize=10)
    for i, mean in enumerate(means):
        ax.plot(x_pos[i], mean, 'ro')
    ax.set_ylabel('Scores')
    ax.set_title('Mean Scores with 95% Confidence Intervals')
    plt.savefig(f'{folder_path}/score_{exp_name}.png')
    plt.show()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = (np.mean(a), stats.sem(a))
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return (m, m - h, m + h)

def main(exp_info_file_path, db_path, exp_name, folder_path):
    with open(exp_info_file_path, 'r') as file:
        exp_info = json.load(file)
    up_result = get_result(exp_info['up_comment_id'], db_path)
    down_result = get_result(exp_info['down_comment_id'], db_path)
    control_result = get_result(exp_info['control_comment_id'], db_path)
    print('up_result:', up_result, 'down_result:', down_result, 'control_result', control_result)
    visualization(up_result, down_result, control_result, exp_name, folder_path)

def get_result(comment_id_lst, db_path):
    db = Database(db_path)
    result_lst = []
    for track_comment_id in comment_id_lst:
        result = db.get_score_comment_id(track_comment_id)
        if result is None:
            print(f'Comment with id:{track_comment_id} not found.')
            result_lst.append(result)
        else:
            result_lst.append(result)
    return result_lst


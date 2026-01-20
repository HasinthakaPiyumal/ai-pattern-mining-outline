# Cluster 0

def get_run_down(dataframe, key, last_steps_ratio=FLAGS.last_steps_ratio):
    dataframe[key + auc_tag] = dataframe.groupby([merged_tag, trial_tag])[key].transform(lambda x: x[int(last_steps_ratio * len(x)):].mean())
    tmp_df = dataframe.groupby([merged_tag, trial_tag]).tail(1)
    run_down = tmp_df.groupby([merged_tag])[key + auc_tag].mean()
    run_down_std = tmp_df.groupby([merged_tag])[key + auc_tag].std()
    run_down_std.name += '_std'
    return (run_down, run_down_std)


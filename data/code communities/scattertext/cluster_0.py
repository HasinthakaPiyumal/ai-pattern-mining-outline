# Cluster 0

def get_log_scale_df(corpus, y_category, x_category):
    term_coord_df = corpus.get_term_freq_df('')
    coord_columns = []
    for category in [y_category, x_category]:
        col_name = category + '_coord'
        term_coord_df[col_name] = np.log(term_coord_df[category] + 1e-06) / np.log(2)
        coord_columns.append(col_name)
    min_offset = term_coord_df[coord_columns].min(axis=0).min()
    for coord_column in coord_columns:
        term_coord_df[coord_column] -= min_offset
    max_offset = term_coord_df[coord_columns].max(axis=0).max()
    for coord_column in coord_columns:
        term_coord_df[coord_column] /= max_offset
    return term_coord_df


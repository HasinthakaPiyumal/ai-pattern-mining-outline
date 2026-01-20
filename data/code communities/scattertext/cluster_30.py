# Cluster 30

def generate_term_metadata(term_struct):
    if p_value_column + '_corr_x' in term_struct:
        x_p = term_struct[p_value_column + '_corr_x']
    elif p_value_column + '_x' in term_struct:
        x_p = term_struct[p_value_column + '_x']
    else:
        x_p = None
    if p_value_column + '_corr_y' in term_struct:
        y_p = term_struct[p_value_column + '_corr_y']
    elif p_value_column + '_y' in term_struct:
        y_p = term_struct[p_value_column + '_y']
    else:
        y_p = None
    if x_p is not None:
        x_p = min(x_p, 1.0 - x_p)
    if y_p is not None:
        y_p = min(y_p, 1.0 - y_p)
    x_d = term_struct[statistic_column + '_x']
    y_d = term_struct[statistic_column + '_y']
    tooltip = '%s: %s: %0.3f' % (x_tooltip_label, statistic_name, x_d)
    if x_p is not None:
        tooltip += '; p: %0.4f' % x_p
    tooltip += '<br/>'
    tooltip += '%s: %s: %0.3f' % (y_tooltip_label, statistic_name, y_d)
    if y_p is not None:
        tooltip += '; p: %0.4f' % y_p
    return {'tooltip': tooltip, 'color': pick_color(x_p, y_p, np.abs(x_d), np.abs(y_d))}

def pick_color(x_pval, y_pval, x_d, y_d):
    if x_d > 0.2 and y_d > 0.2:
        return 'fc00a0'
    if x_d > 0.2 or y_d > 0.2:
        return 'a300fc'
    if x_pval < 0.001 and y_pval < 0.001:
        return 'blue'
    if x_pval < 0.001 or y_pval < 0.001:
        return '00befc'
    else:
        return 'CCCCCC'


# Cluster 0

def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    if true_p_bar != '':
        '\n         Special case to draw in:\n            - green -> TP: True Positives (object detected and matches ground-truth)\n            - red -> FP: False Positives (object detected but does not match ground-truth)\n            - pink -> FN: False Negatives (object not detected but present in the ground-truth)\n        '
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        plt.legend(loc='lower right')
        '\n         Write number on side of bar\n        '
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = ' ' + str(fp_val)
            tp_str_val = fp_str_val + ' ' + str(tp_val)
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == len(sorted_values) - 1:
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        '\n         Write number on side of bar\n        '
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = ' ' + str(val)
            if val < 1.0:
                str_val = ' {0:.2f}'.format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == len(sorted_values) - 1:
                adjust_axes(r, t, fig, axes)
    fig.canvas.set_window_title(window_title)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    '\n     Re-scale height accordingly\n    '
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize='large')
    fig.tight_layout()
    fig.savefig(output_path)
    if to_show:
        plt.show()
    plt.close()

def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


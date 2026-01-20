# Cluster 19

def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]
    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)
    gt_total = iou_5['gt']
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    ap, mrec, mprec = voc_ap(rec[:], prec[:])
    return (ap, mrec, mprec)

def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return (ap, mrec, mpre)

def eval_final_results(result_stat, save_path):
    dump_dict = {}
    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.3)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.5)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.7)
    dump_dict.update({'ap30': ap_30, 'ap_50': ap_50, 'ap_70': ap_70, 'mpre_50': mpre_50, 'mrec_50': mrec_50, 'mpre_70': mpre_70, 'mrec_70': mrec_70})
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
    print('The Average Precision at IOU 0.3 is %.2f, The Average Precision at IOU 0.5 is %.2f, The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70))


# Cluster 1

def log_average_miss_rate(prec, rec, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return (lamr, mr, fppi)
    fppi = 1 - prec
    mr = 1 - rec
    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)
    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return (lamr, mr, fppi)


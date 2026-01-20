# Cluster 111

def binom_confint(k, n, conf_level=0.95, correct=True, alternative='two sided'):
    assert alternative in ('two sided', 'less', 'greater')
    assert np.all(k >= 0) and np.all(k <= n) and np.all(n >= 1)
    assert np.all(conf_level >= 0) and np.all(conf_level <= 1)
    alpha = (1 - conf_level) / 2 if alternative == 'two sided' else 1 - conf_level
    if correct:
        alpha = alpha / len(k)
    alpha = np.array([alpha] * len(k))
    lower = safe_qbeta(alpha, k, n - k + 1)
    upper = safe_qbeta(alpha, k + 1, n - k, lower_tail=False)
    return pd.DataFrame({'lower': lower if alternative in ['two sided', 'greater'] else [0] * len(k), 'upper': upper if alternative in ['two sided', 'less'] else [0] * len(k)})

def safe_qbeta(p, shape1, shape2, lower_tail=True):
    assert len(p) == len(shape1) and len(p) == len(shape2)
    is_0 = shape1 <= 0
    is_1 = shape2 <= 0
    ok = ~(is_0 | is_1)
    x = np.zeros(len(p))
    x[ok] = qbeta(p[ok], shape1[ok], shape2[ok], lower_tail=lower_tail)
    x[is_0 & ~is_1] = 0
    x[is_1 & ~is_0] = 1
    x[is_0 & is_1] = np.nan
    return x


# Cluster 18

def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    vocab = set(p.keys()) | set(q.keys())
    eps = 1e-09

    def _norm(d: Dict[str, float]) -> Dict[str, float]:
        s = sum((d.get(w, 0.0) for w in vocab)) or 1.0
        return {w: (d.get(w, 0.0) + eps) / (s + eps * len(vocab)) for w in vocab}
    P = _norm(p)
    Q = _norm(q)
    M = {w: 0.5 * (P[w] + Q[w]) for w in vocab}

    def _kl(X, Y):
        return sum((X[w] * math.log((X[w] + eps) / (Y[w] + eps)) for w in vocab))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

def _norm(d: Dict[str, float]) -> Dict[str, float]:
    s = sum((d.get(w, 0.0) for w in vocab)) or 1.0
    return {w: (d.get(w, 0.0) + eps) / (s + eps * len(vocab)) for w in vocab}

def _kl(X, Y):
    return sum((X[w] * math.log((X[w] + eps) / (Y[w] + eps)) for w in vocab))


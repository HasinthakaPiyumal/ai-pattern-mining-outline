# Cluster 52

class GaussianKDE(ScipyModel):
    """A wrapper for gaussian Kernel density estimation.

    It was implemented in scipy.stats toolbox. gaussian_kde is slower than statsmodels
    but allows more flexibility.

    When a sample_size is provided the fit method will sample the
    data, and mask the real information. Also, ensure the number of
    entries will be always the value of sample_size.

    Args:
        sample_size(int): amount of parameters to sample
    """
    PARAMETRIC = ParametricType.NON_PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED
    MODEL_CLASS = gaussian_kde

    @store_args
    def __init__(self, sample_size=None, random_state=None, bw_method=None, weights=None):
        self.random_state = validate_random_state(random_state)
        self._sample_size = sample_size
        self.bw_method = bw_method
        self.weights = weights

    def _get_model(self):
        dataset = self._params['dataset']
        self._sample_size = self._sample_size or len(dataset)
        return gaussian_kde(dataset, bw_method=self.bw_method, weights=self.weights)

    def _get_bounds(self):
        X = self._params['dataset']
        lower = np.min(X) - 5 * np.std(X)
        upper = np.max(X) + 5 * np.std(X)
        return (lower, upper)

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._model.evaluate(X)

    @random_state
    def sample(self, n_samples=1):
        """Sample values from this model.

        Argument:
            n_samples (int):
                Number of values to sample

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, 1) with values randomly
                sampled from this model distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._model.resample(size=n_samples)[0]

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        X = np.array(X)
        stdev = np.sqrt(self._model.covariance[0, 0])
        lower = ndtr((self._get_bounds()[0] - self._model.dataset) / stdev)[0]
        uppers = ndtr((X[:, None] - self._model.dataset) / stdev)
        return (uppers - lower).dot(self._model.weights)

    def percent_point(self, U, method='chandrupatla'):
        """Compute the inverse cumulative distribution value for each point in U.

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].
            method (str):
                Whether to use the `chandrupatla` or `bisect` solver.

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        if len(U.shape) > 1:
            raise ValueError(f'Expected 1d array, got {(U,)}.')
        if np.any(U > 1.0) or np.any(U < 0.0):
            raise ValueError('Expected values in range [0.0, 1.0].')
        is_one = U >= 1.0 - EPSILON
        is_zero = U <= EPSILON
        is_valid = ~(is_zero | is_one)
        lower, upper = self._get_bounds()

        def _f(X):
            return self.cumulative_distribution(X) - U[is_valid]
        X = np.zeros(U.shape)
        X[is_one] = float('inf')
        X[is_zero] = float('-inf')
        if is_valid.any():
            lower = np.full(U[is_valid].shape, lower)
            upper = np.full(U[is_valid].shape, upper)
            if method == 'bisect':
                X[is_valid] = bisect(_f, lower, upper)
            else:
                X[is_valid] = chandrupatla(_f, lower, upper)
        return X

    def _fit_constant(self, X):
        sample_size = self._sample_size or len(X)
        constant = np.unique(X)[0]
        self._params = {'dataset': [constant] * sample_size}

    def _fit(self, X):
        if self._sample_size:
            X = gaussian_kde(X, bw_method=self.bw_method, weights=self.weights).resample(self._sample_size)
        self._params = {'dataset': X.tolist()}
        self._model = self._get_model()

    def _is_constant(self):
        return len(np.unique(self._params['dataset'])) == 1

    def _extract_constant(self):
        return self._params['dataset'][0]

    def _set_params(self, params):
        """Set the parameters of this univariate.

        Args:
            params (dict):
                Parameters to recreate this instance.
        """
        self._params = params.copy()
        if self._is_constant():
            constant = self._extract_constant()
            self._set_constant_value(constant)
        else:
            self._model = self._get_model()

def bisect(f, xmin, xmax, tol=1e-08, maxiter=50):
    """Bisection method for finding roots.

    This method implements a simple vectorized routine for identifying
    the root (of a monotonically increasing function) given a bracketing
    interval.

    Arguments:
        f (Callable):
            A function which takes as input a vector x and returns a
            vector with the same number of dimensions.
        xmin (np.ndarray):
            The minimum value for x such that f(x) <= 0.
        xmax (np.ndarray):
            The maximum value for x such that f(x) >= 0.

    Returns:
        numpy.ndarray:
            The value of x such that f(x) is close to 0.
    """
    assert (f(xmin) <= 0.0).all()
    assert (f(xmax) >= 0.0).all()
    for _ in range(maxiter):
        guess = (xmin + xmax) / 2.0
        fguess = f(guess)
        xmin[fguess <= 0] = guess[fguess <= 0]
        xmax[fguess >= 0] = guess[fguess >= 0]
        if (xmax - xmin).max() < tol:
            break
    return (xmin + xmax) / 2.0

def chandrupatla(f, xmin, xmax, eps_m=None, eps_a=None, maxiter=50):
    """Chandrupatla's algorithm.

    This is adapted from [1] which implements Chandrupatla's algorithm [2]
    which starts from a bracketing interval and, conditionally, swaps between
    bisection and inverse quadratic interpolation.

    [1] https://github.com/scipy/scipy/issues/7242#issuecomment-290548427
    [2] https://books.google.com/books?id=cC-8BAAAQBAJ&pg=PA95

    Arguments:
        f (Callable):
            A function which takes as input a vector x and returns a
            vector with the same number of dimensions.
        xmin (np.ndarray):
            The minimum value for x such that f(x) <= 0.
        xmax (np.ndarray):
            The maximum value for x such that f(x) >= 0.

    Returns:
        numpy.ndarray:
            The value of x such that f(x) is close to 0.
    """
    a = xmax
    b = xmin
    fa = f(a)
    fb = f(b)
    shape = np.shape(fa)
    assert shape == np.shape(fb)
    fc = fa
    c = a
    assert (np.sign(fa) * np.sign(fb) <= 0).all()
    t = 0.5
    iqi = np.zeros(shape, dtype=bool)
    eps = np.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2 * eps
    iterations = 0
    terminate = False
    while maxiter > 0:
        maxiter -= 1
        xt = np.clip(a + t * (b - a), xmin, xmax)
        ft = f(xt)
        samesign = np.sign(ft) == np.sign(fa)
        c = np.choose(samesign, [b, a])
        b = np.choose(samesign, [a, b])
        fc = np.choose(samesign, [fb, fa])
        fb = np.choose(samesign, [fa, fb])
        a = xt
        fa = ft
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        xm = np.choose(fa_is_smaller, [b, a])
        fm = np.choose(fa_is_smaller, [fb, fa])
        tol = 2 * eps_m * np.abs(xm) + eps_a
        tlim = tol / np.abs(b - c)
        terminate = np.logical_or(terminate, np.logical_or(fm == 0, tlim > 0.5))
        if np.all(terminate):
            break
        iterations += 1 - terminate
        xi = (a - b) / (c - b)
        phi = (fa - fb) / (fc - fb)
        iqi = np.logical_and(phi ** 2 < xi, (1 - phi) ** 2 < 1 - xi)
        if not shape:
            if iqi:
                eq1 = fa / (fb - fa) * fc / (fb - fc)
                eq2 = (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb)
                t = eq1 + eq2
            else:
                t = 0.5
        else:
            t = np.full(shape, 0.5)
            a2, b2, c2, fa2, fb2, fc2 = (a[iqi], b[iqi], c[iqi], fa[iqi], fb[iqi], fc[iqi])
            t[iqi] = fa2 / (fb2 - fa2) * fc2 / (fb2 - fc2) + (c2 - a2) / (b2 - a2) * fa2 / (fc2 - fa2) * fb2 / (fc2 - fb2)
        t = np.minimum(1 - tlim, np.maximum(tlim, t))
    return xm


# Cluster 58

class Gumbel(Bivariate):
    """Class for clayton copula model."""
    copula_type = CopulaTypes.GUMBEL
    theta_interval = [1, float('inf')]
    invalid_thetas = []

    def generator(self, t):
        """Return the generator function."""
        return np.power(-np.log(t), self.theta)

    def probability_density(self, X):
        """Compute probability density function for given copula family.

        The probability density(PDF) for the Gumbel family of copulas correspond to the formula:

        .. math::

            \\begin{align}
                c(U,V)
                    &= \\frac{\\partial^2 C(u,v)}{\\partial v \\partial u}
                    &= \\frac{C(u,v)}{uv} \\frac{((-\\ln u)^{\\theta}  # noqa: JS101
                    + (-\\ln v)^{\\theta})^{\\frac{2}  # noqa: JS101
                {\\theta} - 2 }}{(\\ln u \\ln v)^{1 - \\theta}}  # noqa: JS101
                ( 1 + (\\theta-1) \\big((-\\ln u)^\\theta
                + (-\\ln v)^\\theta\\big)^{-1/\\theta})
            \\end{align}

        Args:
            X (numpy.ndarray)

        Returns:
            numpy.ndarray

        """
        self.check_fit()
        U, V = split_matrix(X)
        if self.theta == 1:
            return U * V
        else:
            a = np.power(U * V, -1)
            tmp = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
            b = np.power(tmp, -2 + 2.0 / self.theta)
            c = np.power(np.log(U) * np.log(V), self.theta - 1)
            d = 1 + (self.theta - 1) * np.power(tmp, -1.0 / self.theta)
            return self.cumulative_distribution(X) * a * b * c * d

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution function for the Gumbel copula.

        The cumulative density(cdf), or distribution function for the Gumbel family of copulas
        correspond to the formula:

        .. math:: C(u,v) = e^{-((-\\ln u)^{\\theta} + (-\\ln v)^{\\theta})^{\\frac{1}{\\theta}}}

        Args:
            X (np.ndarray)

        Returns:
            np.ndarray: cumulative probability for the given datapoints, cdf(X).

        """
        self.check_fit()
        U, V = split_matrix(X)
        if self.theta == 1:
            return U * V
        else:
            h = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
            h = -np.power(h, 1.0 / self.theta)
            cdfs = np.exp(h)
            return cdfs

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^{-1}`.

        Args:
            y (np.ndarray): value of :math:`C(u|v)`.
            v (np.ndarray): given value of v.

        """
        self.check_fit()
        if self.theta == 1:
            return y
        else:
            return super().percent_point(y, V)

    def partial_derivative(self, X):
        """Compute partial derivative of cumulative distribution.

        The partial derivative of the copula(CDF) is the conditional CDF.

        .. math:: F(v|u) = \\frac{\\partial C(u,v)}{\\partial u} =
            C(u,v)\\frac{((-\\ln u)^{\\theta} + (-\\ln v)^{\\theta})^{\\frac{1}{\\theta} - 1}}
            {\\theta(- \\ln u)^{1 -\\theta}}

        Args:
            X (np.ndarray)
            y (float)

        Returns:
            numpy.ndarray

        """
        self.check_fit()
        U, V = split_matrix(X)
        if self.theta == 1:
            return V
        else:
            t1 = np.power(-np.log(U), self.theta)
            t2 = np.power(-np.log(V), self.theta)
            p1 = self.cumulative_distribution(X)
            p2 = np.power(t1 + t2, -1 + 1.0 / self.theta)
            p3 = np.power(-np.log(V), self.theta - 1)
            return p1 * p2 * p3 / V

    def compute_theta(self):
        """Compute theta parameter using Kendall's tau.

        On Gumbel copula :math:`\\tau` is defined as :math:`τ = \\frac{θ−1}{θ}`
        that we solve as :math:`θ = \\frac{1}{1-τ}`
        """
        if self.tau == 1:
            raise ValueError("Tau value can't be 1")
        return 1 / (1 - self.tau)

def split_matrix(X):
    """Split an (n,2) numpy.array into two vectors.

    Args:
        X(numpy.array): Matrix of shape (n,2)

    Returns:
        tuple[numpy.array]: Both of shape (n,)

    """
    if len(X):
        return (X[:, 0], X[:, 1])
    return (np.array([]), np.array([]))

class Bivariate(object):
    """Base class for bivariate copulas.

    This class allows to instantiate all its subclasses and serves as a unique entry point for
    the bivariate copulas classes.

    >>> Bivariate(copula_type=CopulaTypes.FRANK).__class__
    copulas.bivariate.frank.Frank

    >>> Bivariate(copula_type='frank').__class__
    copulas.bivariate.frank.Frank


    Args:
        copula_type (Union[CopulaType, str]): Subtype of the copula.
        random_state (Union[int, np.random.RandomState, None]): Seed or RandomState
            for the random generator.

    Attributes:
        copula_type(CopulaTypes): Family of the copula a subclass belongs to.
        _subclasses(list[type]): List of declared subclasses.
        theta_interval(list[float]): Interval of valid thetas for the given copula family.
        invalid_thetas(list[float]): Values that, even though they belong to
            :attr:`theta_interval`, shouldn't be considered valid.
        tau (float): Kendall's tau for the data given at :meth:`fit`.
        theta(float): Parameter for the copula.

    """
    copula_type = None
    _subclasses = []
    theta_interval = []
    invalid_thetas = []
    theta = None
    tau = None

    @classmethod
    def _get_subclasses(cls):
        """Find recursively subclasses for the current class object.

        Returns:
            list[Bivariate]: List of subclass objects.

        """
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(subclass._get_subclasses())
        return subclasses

    @classmethod
    def subclasses(cls):
        """Return a list of subclasses for the current class object.

        Returns:
            list[Bivariate]: Subclasses for given class.

        """
        if not cls._subclasses:
            cls._subclasses = cls._get_subclasses()
        return cls._subclasses

    def __new__(cls, *args, **kwargs):
        """Create and return a new object.

        Returns:
            Bivariate: New object.
        """
        copula_type = kwargs.get('copula_type', None)
        if copula_type is None:
            return super(Bivariate, cls).__new__(cls)
        if not isinstance(copula_type, CopulaTypes):
            if isinstance(copula_type, str) and copula_type.upper() in CopulaTypes.__members__:
                copula_type = CopulaTypes[copula_type.upper()]
            else:
                raise ValueError(f'Invalid copula type {copula_type}')
        for subclass in cls.subclasses():
            if subclass.copula_type is copula_type:
                return super(Bivariate, cls).__new__(subclass)

    def __init__(self, copula_type=None, random_state=None):
        """Initialize Bivariate object.

        Args:
            copula_type (CopulaType or str): Subtype of the copula.
            random_state (int, np.random.RandomState, or None): Seed or RandomState
                for the random generator.
        """
        self.random_state = validate_random_state(random_state)

    def check_theta(self):
        """Validate the computed theta against the copula specification.

        This method is used to assert the computed theta is in the valid range for the copula.

        Raises:
            ValueError: If theta is not in :attr:`theta_interval` or is in :attr:`invalid_thetas`,

        """
        lower, upper = self.theta_interval
        if not lower <= self.theta <= upper or self.theta in self.invalid_thetas:
            message = 'The computed theta value {} is out of limits for the given {} copula.'
            raise ValueError(message.format(self.theta, self.copula_type.name))

    def check_fit(self):
        """Assert that the model is fit and the computed `theta` is valid.

        Raises:
            NotFittedError: if the model is  not fitted.
            ValueError: if the computed theta is invalid.

        """
        if not self.theta:
            raise NotFittedError('This model is not fitted.')
        self.check_theta()

    def check_marginal(self, u):
        """Check that the marginals are uniformly distributed.

        Args:
            u(np.ndarray): Array of datapoints with shape (n,).

        Raises:
            ValueError: If the data does not appear uniformly distributed.
        """
        if min(u) < 0.0 or max(u) > 1.0:
            raise ValueError('Marginal value out of bounds.')
        emperical_cdf = np.sort(u)
        uniform_cdf = np.linspace(0.0, 1.0, num=len(u))
        ks_statistic = max(np.abs(emperical_cdf - uniform_cdf))
        if ks_statistic > 1.627 / np.sqrt(len(u)):
            warnings.warn('Data does not appear to be uniform.', category=RuntimeWarning)

    def _compute_theta(self):
        """Compute theta, validate it and assign it to self."""
        self.theta = self.compute_theta()
        self.check_theta()

    def fit(self, X):
        """Fit a model to the data updating the parameters.

        Args:
            X(np.ndarray): Array of datapoints with shape (n,2).

        Return:
            None
        """
        U, V = split_matrix(X)
        self.check_marginal(U)
        self.check_marginal(V)
        self.tau = stats.kendalltau(U, V)[0]
        if np.isnan(self.tau):
            if len(np.unique(U)) == 1 or len(np.unique(V)) == 1:
                raise ValueError('Constant column.')
            raise ValueError('Unable to compute tau.')
        self._compute_theta()

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this object.

        Returns:
            dict: Parameters of the copula.

        """
        return {'copula_type': self.copula_type.name, 'theta': self.theta, 'tau': self.tau}

    @classmethod
    def from_dict(cls, copula_dict):
        """Create a new instance from the given parameters.

        Args:
            copula_dict: `dict` with the parameters to replicate the copula.
              Like the output of `Bivariate.to_dict`

        Returns:
            Bivariate: Instance of the copula defined on the parameters.

        """
        instance = cls(copula_type=copula_dict['copula_type'])
        instance.theta = copula_dict['theta']
        instance.tau = copula_dict['tau']
        return instance

    def infer(self, X):
        """Take in subset of values and predicts the rest."""
        raise NotImplementedError

    def generator(self, t):
        """Compute the generator function for Archimedian copulas.

        The generator is a function
        :math:`\\psi: [0,1]\\times\\Theta \\rightarrow [0, \\infty)`  # noqa: JS101

        that given an Archimedian copula fulfills:
        .. math:: C(u,v) = \\psi^{-1}(\\psi(u) + \\psi(v))


        In a more generic way:

        .. math:: C(u_1, u_2, ..., u_n;\\theta) = \\psi^-1(\\sum_0^n{\\psi(u_i;\\theta)}; \\theta)

        """
        raise NotImplementedError

    def probability_density(self, X):
        """Compute probability density function for given copula family.

        The probability density(pdf) for a given copula is defined as:

        .. math:: c(U,V) = \\frac{\\partial^2 C(u,v)}{\\partial v \\partial u}

        Args:
            X(np.ndarray): Shape (n, 2).Datapoints to compute pdf.

        Returns:
            np.array: Probability density for the input values.

        """
        raise NotImplementedError

    def log_probability_density(self, X):
        """Return log probability density of model.

        The log probability should be overridden with numerically stable
        variants whenever possible.

        Arguments:
            X: `np.ndarray` of shape (n, 1).

        Returns:
            np.ndarray

        """
        return np.log(self.probability_density(X))

    def pdf(self, X):
        """Shortcut to :meth:`probability_density`."""
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution function for the copula, :math:`C(u, v)`.

        Args:
            X(np.ndarray):

        Returns:
            numpy.array: cumulative probability

        """
        raise NotImplementedError

    def cdf(self, X):
        """Shortcut to :meth:`cumulative_distribution`."""
        return self.cumulative_distribution(X)

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^{-1}`.

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """
        self.check_fit()
        result = []
        for _y, _v in zip(y, V):

            def f(u):
                return self.partial_derivative_scalar(u, _v) - _y
            minimum = brentq(f, EPSILON, 1.0)
            if isinstance(minimum, np.ndarray):
                minimum = minimum[0]
            result.append(minimum)
        return np.array(result)

    def ppf(self, y, V):
        """Shortcut to :meth:`percent_point`."""
        return self.percent_point(y, V)

    def partial_derivative(self, X):
        """Compute partial derivative of cumulative distribution.

        The partial derivative of the copula(CDF) is the conditional CDF.

         .. math:: F(v|u) = \\frac{\\partial C(u,v)}{\\partial u}

        The base class provides a finite difference approximation of the
        partial derivative of the CDF with respect to u.

        Args:
            X(np.ndarray)
            y(float)

        Returns:
            np.ndarray

        """
        delta = -2 * (X[:, 1] > 0.5) + 1
        delta = 0.0001 * delta
        X_prime = X.copy()
        X_prime[:, 1] += delta
        f = self.cumulative_distribution(X)
        f_prime = self.cumulative_distribution(X_prime)
        return (f_prime - f) / delta

    def partial_derivative_scalar(self, U, V):
        """Compute partial derivative :math:`C(u|v)` of cumulative density of single values."""
        self.check_fit()
        X = np.column_stack((U, V))
        return self.partial_derivative(X)

    def set_random_state(self, random_state):
        """Set the random state.

        Args:
            random_state (int, np.random.RandomState, or None): Seed or RandomState
                for the random generator.
        """
        self.random_state = validate_random_state(random_state)

    @random_state
    def sample(self, n_samples):
        """Generate specified `n_samples` of new data from model.

        The sampled are generated using the inverse transform method `v~U[0,1],v~C^-1(u|v)`

        Args:
            n_samples (int): amount of samples to create.

        Returns:
            np.ndarray: Array of length `n_samples` with generated data from the model.

        """
        if self.tau > 1 or self.tau < -1:
            raise ValueError('The range for correlation measure is [-1,1].')
        v = np.random.uniform(0, 1, n_samples)
        c = np.random.uniform(0, 1, n_samples)
        u = self.percent_point(c, v)
        return np.column_stack((u, v))

    def compute_theta(self):
        """Compute theta parameter using Kendall's tau."""
        raise NotImplementedError

    @classmethod
    def select_copula(cls, X):
        """Select best copula function based on likelihood.

        Given out candidate copulas the procedure proposed for selecting the one
        that best fit to a dataset of pairs :math:`\\{(u_j, v_j )\\}, j=1,2,...n` , is as follows:

        1. Estimate the most likely parameter :math:`\\theta` of each copula candidate for the given
           dataset.

        2. Construct :math:`R(z|\\theta)`. Calculate the area under the tail for each of the copula
           candidates.

        3. Compare the areas: :math:`a_u` achieved using empirical copula against the ones
           achieved for the copula candidates. Score the outcome of the comparison from 3 (best)
           down to 1 (worst).

        4. Proceed as in steps 2- 3 with the lower tail and function :math:`L`.

        5. Finally the sum of empirical upper and lower tail functions is compared against
           :math:`R + L`. Scores of the three comparisons are summed and the candidate with the
           highest value is selected.

        Args:
            X(np.ndarray): Matrix of shape (n,2).

        Returns:
            copula: Best copula that fits for it.

        """
        from sdgx.models.components.sdv_copulas.bivariate import select_copula
        warnings.warn('`Bivariate.select_copula` has been deprecated and will be removed in a later release. Please use `copulas.bivariate.select_copula` instead', DeprecationWarning)
        return select_copula(X)

    def save(self, filename):
        """Save the internal state of a copula in the specified filename.

        Args:
            filename(str): Path to save.

        Returns:
            None

        """
        content = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(content, f)

    @classmethod
    def load(cls, copula_path):
        """Create a new instance from a file.

        Args:
            copula_path(str): Path to file with the serialized copula.

        Returns:
            Bivariate: Instance with the parameters stored in the file.

        """
        with open(copula_path) as f:
            copula_dict = json.load(f)
        return cls.from_dict(copula_dict)

class Independence(Bivariate):
    """This class represent the copula for two independent variables."""
    copula_type = CopulaTypes.INDEPENDENCE

    def fit(self, X):
        """Fit the copula to the given data.

        Args:
            X (numpy.array): Probabilites in a matrix shaped (n, 2)

        Returns:
            None

        """

    def generator(self, t):
        """Compute the generator function for the Copula.

        The generator function is a function f(t), such that an archimedian copula can be
        defined as

        C(u1, ..., uN) = f(f^-1(u1), ..., f^-1(uN)).

        Args:
            t(numpy.array)

        Returns:
            np.array

        """
        return np.log(t)

    def probability_density(self, X):
        """Compute the probability density for the independence copula."""
        return np.all((0.0 <= X) & (X <= 1.0), axis=1).astype(float)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution of the independence bivariate copula is the product.

        Args:
            X(numpy.array): Matrix of shape (n,2), whose values are in [0, 1]

        Returns:
            numpy.array: Cumulative distribution values of given input.

        """
        U, V = split_matrix(X)
        return U * V

    def partial_derivative(self, X):
        """Compute the conditional probability of one event conditiones to the other.

        In the case of the independence copula, due to C(u,v) = u*v, we have that
        F(u|v) = dC/du = v.

        Args:
            X()

        """
        _, V = split_matrix(X)
        return V

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`F(u|v)^-1`.

        Args:
            y: `np.ndarray` value of :math:`F(u|v)`.
            v: `np.ndarray` given value of v.

        """
        self.check_fit()
        return y

def _compute_empirical(X):
    """Compute empirical distribution.

    Args:
        X(numpy.array): Shape (n,2); Datapoints to compute the empirical(frequentist) copula.

    Return:
        tuple(list):

    """
    z_left = []
    z_right = []
    L = []
    R = []
    U, V = split_matrix(X)
    N = len(U)
    base = np.linspace(EPSILON, 1.0 - EPSILON, COMPUTE_EMPIRICAL_STEPS)
    for k in range(COMPUTE_EMPIRICAL_STEPS):
        left = sum(np.logical_and(U <= base[k], V <= base[k])) / N
        right = sum(np.logical_and(U >= base[k], V >= base[k])) / N
        if left > 0:
            z_left.append(base[k])
            L.append(left / base[k] ** 2)
        if right > 0:
            z_right.append(base[k])
            R.append(right / (1 - z_right[k]) ** 2)
    return (z_left, L, z_right, R)

def _compute_candidates(copulas, left_tail, right_tail):
    """Compute dependencies.

    Args:
        copulas(list[Bivariate]): Fitted instances of bivariate copulas.
        z_left(list):
        z_right(list):

    Returns:
        tuple[list]: Arrays of left and right dependencies for the empirical copula.


    """
    left = []
    right = []
    X_left = np.column_stack((left_tail, left_tail))
    X_right = np.column_stack((right_tail, right_tail))
    for copula in copulas:
        left.append(copula.cumulative_distribution(X_left) / np.power(left_tail, 2))
        right.append(_compute_tail(copula.cumulative_distribution(X_right), right_tail))
    return (left, right)

def _compute_tail(c, z):
    """Compute upper concentration function for tail.

    The upper tail concentration function is defined by:

    .. math:: R(z) = \\frac{[1 − 2z + C(z, z)]}{(1 − z)^{2}}

    Args:
        c(Iterable): Values of :math:`C(z,z)`.
        z(Iterable): Values for the empirical copula.

    Returns:
        numpy.ndarray

    """
    return (1.0 - 2 * np.asarray(z) + c) / np.power(1.0 - np.asarray(z), 2)

def select_copula(X):
    """Select best copula function based on likelihood.

    Given out candidate copulas the procedure proposed for selecting the one
    that best fit to a dataset of pairs :math:`\\{(u_j, v_j )\\}, j=1,2,...n` , is as follows:

    1. Estimate the most likely parameter :math:`\\theta` of each copula candidate for the given
       dataset.

    2. Construct :math:`R(z|\\theta)`. Calculate the area under the tail for each of the copula
       candidates.

    3. Compare the areas: :math:`a_u` achieved using empirical copula against the ones
       achieved for the copula candidates. Score the outcome of the comparison from 3 (best)
       down to 1 (worst).

    4. Proceed as in steps 2- 3 with the lower tail and function :math:`L`.

    5. Finally the sum of empirical upper and lower tail functions is compared against
       :math:`R + L`. Scores of the three comparisons are summed and the candidate with the
       highest value is selected.

    Args:
        X(np.ndarray): Matrix of shape (n,2).

    Returns:
        copula: Best copula that fits for it.

    """
    frank = Frank()
    frank.fit(X)
    if frank.tau <= 0:
        return frank
    copula_candidates = [frank]
    for copula_class in [Clayton, Gumbel]:
        try:
            copula = copula_class()
            copula.tau = frank.tau
            copula._compute_theta()
            copula_candidates.append(copula)
        except ValueError:
            pass
    left_tail, empirical_left_aut, right_tail, empirical_right_aut = _compute_empirical(X)
    candidate_left_auts, candidate_right_auts = _compute_candidates(copula_candidates, left_tail, right_tail)
    empirical_aut = np.concatenate((empirical_left_aut, empirical_right_aut))
    candidate_auts = [np.concatenate((left, right)) for left, right in zip(candidate_left_auts, candidate_right_auts)]
    diff_left = [np.sum((empirical_left_aut - left) ** 2) for left in candidate_left_auts]
    diff_right = [np.sum((empirical_right_aut - right) ** 2) for right in candidate_right_auts]
    diff_both = [np.sum((empirical_aut - candidate) ** 2) for candidate in candidate_auts]
    score_left = pd.Series(diff_left).rank(ascending=False)
    score_right = pd.Series(diff_right).rank(ascending=False)
    score_both = pd.Series(diff_both).rank(ascending=False)
    score = score_left + score_right + score_both
    selected_copula = np.argmax(score.to_numpy())
    return copula_candidates[selected_copula]

class Frank(Bivariate):
    """Class for Frank copula model."""
    copula_type = CopulaTypes.FRANK
    theta_interval = [-float('inf'), float('inf')]
    invalid_thetas = [0]

    def generator(self, t):
        """Return the generator function."""
        a = (np.exp(-self.theta * t) - 1) / (np.exp(-self.theta) - 1)
        return -np.log(a)

    def _g(self, z):
        """Assist in solving the Frank copula.

        This functions encapsulates :math:`g(z) = e^{-\\theta z} - 1` used on Frank copulas.

        Argument:
            z: np.ndarray

        Returns:
            np.ndarray

        """
        return np.exp(-self.theta * z) - 1

    def probability_density(self, X):
        """Compute probability density function for given copula family.

        The probability density(PDF) for the Frank family of copulas correspond to the formula:

        .. math:: c(U,V) = \\frac{\\partial^2 C(u,v)}{\\partial v \\partial u} =
             \\frac{-\\theta g(1)(1 + g(u + v))}{(g(u) g(v) + g(1)) ^ 2}

        Where the g function is defined by:

        .. math:: g(x) = e^{-\\theta x} - 1

        Args:
            X: `np.ndarray`

        Returns:
            np.array: probability density

        """
        self.check_fit()
        U, V = split_matrix(X)
        if self.theta == 0:
            return U * V
        else:
            num = -self.theta * self._g(1) * (1 + self._g(U + V))
            aux = self._g(U) * self._g(V) + self._g(1)
            den = np.power(aux, 2)
            return num / den

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution function for the Frank copula.

        The cumulative density(cdf), or distribution function for the Frank family of copulas
        correspond to the formula:

        .. math:: C(u,v) =  −\\frac{\\ln({\\frac{1 + g(u) g(v)}{g(1)}})}{\\theta}


        Args:
            X: `np.ndarray`

        Returns:
            np.array: cumulative distribution

        """
        self.check_fit()
        U, V = split_matrix(X)
        num = (np.exp(-self.theta * U) - 1) * (np.exp(-self.theta * V) - 1)
        den = np.exp(-self.theta) - 1
        return -1.0 / self.theta * np.log(1 + num / den)

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^{-1}`.

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """
        self.check_fit()
        if self.theta == 0:
            return V
        else:
            return super().percent_point(y, V)

    def partial_derivative(self, X):
        """Compute partial derivative of cumulative distribution.

        The partial derivative of the copula(CDF) is the conditional CDF.

        .. math:: F(v|u) = \\frac{\\partial}{\\partial u}C(u,v) =
            \\frac{g(u)g(v) + g(v)}{g(u)g(v) + g(1)}

        Args:
            X (np.ndarray)
            y (float)

        Returns:
            np.ndarray

        """
        self.check_fit()
        U, V = split_matrix(X)
        if self.theta == 0:
            return V
        else:
            num = self._g(U) * self._g(V) + self._g(U)
            den = self._g(U) * self._g(V) + self._g(1)
            return num / den

    def compute_theta(self):
        """Compute theta parameter using Kendall's tau.

        On Frank copula, the relationship between tau and theta is defined by:

        .. math:: \\tau = 1 − \\frac{4}{\\theta} + \\frac{4}{\\theta^2}\\int_0^\\theta \\!
            \\frac{t}{e^t -1} \\mathrm{d}t.

        In order to solve it, we can simplify it as

        .. math:: 0 = 1 + \\frac{4}{\\theta}(D_1(\\theta) - 1) - \\tau

        where the function D is the Debye function of first order, defined as:

        .. math:: D_1(x) = \\frac{1}{x}\\int_0^x\\frac{t}{e^t -1} \\mathrm{d}t.

        """
        result = least_squares(self._tau_to_theta, 1, bounds=(MIN_FLOAT_LOG, MAX_FLOAT_LOG))
        return result.x[0]

    def _tau_to_theta(self, alpha):
        """Relationship between tau and theta as a solvable equation."""

        def debye(t):
            return t / (np.exp(t) - 1)
        debye_value = integrate.quad(debye, EPSILON, alpha)[0] / alpha
        return 4 * (debye_value - 1) / alpha + 1 - self.tau

class Clayton(Bivariate):
    """Class for clayton copula model."""
    copula_type = CopulaTypes.CLAYTON
    theta_interval = [0, float('inf')]
    invalid_thetas = []

    def generator(self, t):
        """Compute the generator function for Clayton copula family.

        The generator is a function
        :math:`\\psi: [0,1]\\times\\Theta \\rightarrow [0, \\infty)`  # noqa: JS101

        that given an Archimedian copula fulfills:
        .. math:: C(u,v) = \\psi^{-1}(\\psi(u) + \\psi(v))

        Args:
            t (numpy.ndarray)

        Returns:
            numpy.ndarray

        """
        self.check_fit()
        return 1.0 / self.theta * (np.power(t, -self.theta) - 1)

    def probability_density(self, X):
        """Compute probability density function for given copula family.

        The probability density(PDF) for the Clayton family of copulas correspond to the formula:

        .. math:: c(U,V) = \\frac{\\partial^2}{\\partial v \\partial u}C(u,v) =
            (\\theta + 1)(uv)^{-\\theta-1}(u^{-\\theta} +
            v^{-\\theta} - 1)^{-\\frac{2\\theta + 1}{\\theta}}

        Args:
            X (numpy.ndarray)

        Returns:
            numpy.ndarray: Probability density for the input values.

        """
        self.check_fit()
        U, V = split_matrix(X)
        a = (self.theta + 1) * np.power(U * V, -(self.theta + 1))
        b = np.power(U, -self.theta) + np.power(V, -self.theta) - 1
        c = -(2 * self.theta + 1) / self.theta
        return a * np.power(b, c)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution function for the clayton copula.

        The cumulative density(cdf), or distribution function for the Clayton family of copulas
        correspond to the formula:

        .. math:: C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}

        Args:
            X (numpy.ndarray)

        Returns:
            numpy.ndarray: cumulative probability.

        """
        self.check_fit()
        U, V = split_matrix(X)
        if (V == 0).all() or (U == 0).all():
            return np.zeros(V.shape[0])
        else:
            cdfs = [np.power(np.power(U[i], -self.theta) + np.power(V[i], -self.theta) - 1, -1.0 / self.theta) if U[i] > 0 and V[i] > 0 else 0 for i in range(len(U))]
            return np.array(cdfs)

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^{-1}`.

        Args:
            y (numpy.ndarray): Value of :math:`C(u|v)`.
            v (numpy.ndarray): given value of v.
        """
        self.check_fit()
        if self.theta < 0:
            return V
        else:
            a = np.power(y, self.theta / (-1 - self.theta))
            b = np.power(V, self.theta)
            if (b == 0).all():
                return np.ones(len(V))
            return np.power((a + b - 1) / b, -1 / self.theta)

    def partial_derivative(self, X):
        """Compute partial derivative of cumulative distribution.

        The partial derivative of the copula(CDF) is the conditional CDF.

        .. math:: F(v|u) = \\frac{\\partial C(u,v)}{\\partial u} =
            u^{- \\theta - 1}(u^{-\\theta} + v^{-\\theta} - 1)^{-\\frac{\\theta+1}{\\theta}}

        Args:
            X (np.ndarray)
            y (float)

        Returns:
            numpy.ndarray: Derivatives

        """
        self.check_fit()
        U, V = split_matrix(X)
        A = np.power(V, -self.theta - 1)
        if (A == np.inf).any():
            return np.zeros(len(V))
        B = np.power(V, -self.theta) + np.power(U, -self.theta) - 1
        h = np.power(B, (-1 - self.theta) / self.theta)
        return A * h

    def compute_theta(self):
        """Compute theta parameter using Kendall's tau.

        On Clayton copula this is

        .. math:: τ = θ/(θ + 2) \\implies θ = 2τ/(1-τ)
        .. math:: θ ∈ (0, ∞)

        On the corner case of :math:`τ = 1`, return infinite.
        """
        if self.tau == 1:
            return np.inf
        return 2 * self.tau / (1 - self.tau)


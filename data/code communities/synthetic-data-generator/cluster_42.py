# Cluster 42

class RandomIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of random integers with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomIntegerGenerator.generate(num_rows).astype(float))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 2e-05, 'memory': 400.0}, 'transform': {'time': 5e-05, 'memory': 1000.0}, 'reverse_transform': {'time': 1e-05, 'memory': 1000.0}}

def add_nans(array):
    """Add a random amount of NaN values to the given array.

    Args:
        array (np.array):
            1 dimensional numpy array.

    Returns:
        np.array:
            The same array with some values replaced by NaNs.
    """
    if array.dtype.kind == 'i':
        array = array.astype(float)
    length = len(array)
    num_nulls = np.random.randint(1, length)
    nulls = np.random.choice(range(length), num_nulls)
    array[nulls] = np.nan
    return array

class RandomStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of random strings with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomStringGenerator.generate(num_rows).astype('O'))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 2e-05, 'memory': 400.0}, 'transform': {'time': 1e-05, 'memory': 1000.0}, 'reverse_transform': {'time': 1e-05, 'memory': 1000.0}}

class SingleIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array with a single integer with some nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(SingleIntegerGenerator.generate(num_rows).astype(float))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 2e-05, 'memory': 400.0}, 'transform': {'time': 3e-05, 'memory': 400.0}, 'reverse_transform': {'time': 1e-05, 'memory': 500.0}}

class SingleStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of a single string with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(SingleStringGenerator.generate(num_rows).astype('O'))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 2e-05, 'memory': 400.0}, 'transform': {'time': 3e-05, 'memory': 400.0}, 'reverse_transform': {'time': 1e-05, 'memory': 500.0}}

class UniqueIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of unique integers with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(UniqueIntegerGenerator.generate(num_rows))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 2e-05, 'memory': 1000.0}, 'transform': {'time': 0.0002, 'memory': 1000000.0}, 'reverse_transform': {'time': 0.0002, 'memory': 1000000.0}}

class UniqueStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of unique strings with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(UniqueStringGenerator.generate(num_rows).astype('O'))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 2e-05, 'memory': 1000.0}, 'transform': {'time': 0.0005, 'memory': 1000000.0}, 'reverse_transform': {'time': 0.0002, 'memory': 1000000.0}}

class RandomStringNaNsGenerator(PIIGenerator):
    """Generator that creates an array of random strings with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomStringGenerator.generate(num_rows).astype('O'))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 1e-05, 'memory': 400.0}, 'transform': {'time': 1e-05, 'memory': 1000.0}, 'reverse_transform': {'time': 2e-05, 'memory': 1000.0}}

class RandomIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of random integers with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomIntegerGenerator.generate(num_rows).astype(float))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 0.001, 'memory': 2500.0}, 'transform': {'time': 4e-05, 'memory': 400.0}, 'reverse_transform': {'time': 2e-05, 'memory': 300.0}}

class ConstantIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates a constant array with a random integer with some nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(ConstantIntegerGenerator.generate(num_rows).astype(float))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 0.001, 'memory': 600.0}, 'transform': {'time': 3e-05, 'memory': 400.0}, 'reverse_transform': {'time': 2e-05, 'memory': 300.0}}

class AlmostConstantIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates an array with 2 only values, one of them repeated, and NaNs."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        ii32 = np.iinfo(np.int32)
        values = np.random.randint(ii32.min, ii32.max, size=2)
        additional_values = np.full(num_rows - 2, values[1]).astype(float)
        array = np.concatenate([values, add_nans(additional_values)])
        np.random.shuffle(array)
        return array

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 0.001, 'memory': 2500.0}, 'transform': {'time': 3e-05, 'memory': 1000.0}, 'reverse_transform': {'time': 2e-05, 'memory': 1000.0}}

class NormalNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of normally distributed float values, with NaNs."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(NormalGenerator.generate(num_rows))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 0.001, 'memory': 2500.0}, 'transform': {'time': 4e-05, 'memory': 400.0}, 'reverse_transform': {'time': 5e-05, 'memory': 300.0}}

class BigNormalNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of normally distributed float values, with NaNs."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(BigNormalGenerator.generate(num_rows))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 0.001, 'memory': 2500.0}, 'transform': {'time': 3e-05, 'memory': 400.0}, 'reverse_transform': {'time': 2e-05, 'memory': 300.0}}

class RandomStringNaNsGenerator(RegexGeneratorGenerator):
    """Generator that creates an array of random strings with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomStringGenerator.generate(num_rows).astype('O'))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 1e-05, 'memory': 400.0}, 'transform': {'time': 1e-05, 'memory': 1000.0}, 'reverse_transform': {'time': 2e-05, 'memory': 1000.0}}

class RandomGapDatetimeNaNsGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps and NaNs."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        dates = RandomGapDatetimeGenerator.generate(num_rows)
        return add_nans(dates.astype('O'))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 5e-05, 'memory': 500.0}, 'transform': {'time': 5e-05, 'memory': 1000.0}, 'reverse_transform': {'time': 5e-05, 'memory': 1000.0}}


# Cluster 43

class RandomMixedGenerator(CategoricalGenerator):
    """Generator that creates an array of random mixed types.

    Mixed types include: int, float, bool, string, datetime.
    """

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        cat_size = 5
        categories = np.hstack([cat.astype('O') for cat in [RandomGapDatetimeGenerator.generate(cat_size), np.random.randint(0, 100, cat_size), np.random.uniform(0, 100, cat_size), np.arange(cat_size).astype(str), np.array([True, False])]])
        return np.random.choice(a=categories, size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {'fit': {'time': 2e-05, 'memory': 400.0}, 'transform': {'time': 1e-05, 'memory': 1000.0}, 'reverse_transform': {'time': 1e-05, 'memory': 2000.0}}


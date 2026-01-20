# Cluster 16

class Indicator:
    """Class representing an indicator.

    Args:
        name: Name of indicator.
        fn: :class:`Callable` used to compute the series of indicator values.
        kwargs: ``dict`` of kwargs to pass to ``fn``.
    """

    def __init__(self, name: str, fn: Callable[..., NDArray[np.float64]], kwargs: dict[str, Any]):
        self.name = name
        self._fn = functools.partial(fn, **kwargs)
        self._kwargs = kwargs

    def relative_entropy(self, data: Union[BarData, pd.DataFrame]) -> float:
        """Generates indicator data with ``data`` and computes its relative
        `entropy
        <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.
        """
        return relative_entropy(self(data).values)

    def iqr(self, data: Union[BarData, pd.DataFrame]) -> float:
        """Generates indicator data with ``data`` and computes its
        `interquartile range (IQR)
        <https://en.wikipedia.org/wiki/Interquartile_range>`_.
        """
        return iqr(self(data).values)

    def __call__(self, data: Union[BarData, pd.DataFrame]) -> pd.Series:
        """Computes indicator values."""
        if isinstance(data, pd.DataFrame):
            data = _to_bar_data(data)
        values = self._fn(data)
        if isinstance(values, pd.Series):
            values = values.to_numpy()
        if len(values.shape) != 1:
            raise ValueError(f'Indicator {self.name} must return a one-dimensional array.')
        return pd.Series(values, index=data.date)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Indicator({self.name!r}, {self._kwargs})'

def _to_bar_data(df: pd.DataFrame) -> BarData:
    df = df.reset_index()
    required_cols = (DataCol.DATE, DataCol.OPEN, DataCol.HIGH, DataCol.LOW, DataCol.CLOSE)
    for col in required_cols:
        if col.value not in df.columns:
            raise ValueError(f'DataFrame is missing required column: {col.value}')
    return BarData(**{col.value: df[col.value].to_numpy() for col in required_cols}, **{col.value: df[col.value].to_numpy() if col.value in df.columns else None for col in (DataCol.VOLUME, DataCol.VWAP)}, **{col: df[col].to_numpy() if col in df.columns else None for col in StaticScope.instance().custom_data_cols})

@pytest.mark.usefixtures('setup_teardown')
def test_to_bar_data(scope, data_source_df):
    bar_data = _to_bar_data(data_source_df)
    for col in scope.all_data_cols:
        expected = data_source_df[col].to_numpy() if col in data_source_df.columns else None
        assert np.array_equal(getattr(bar_data, col), expected)

@pytest.mark.parametrize('drop_col', ['date', 'open', 'high', 'low', 'close'])
def test_to_bar_data_when_missing_cols_then_error(drop_col, data_source_df):
    with pytest.raises(ValueError, match=f'DataFrame is missing required column: {drop_col}'):
        _to_bar_data(data_source_df.drop(columns=drop_col))


# Cluster 26

def test_register_columns(scope):
    scope.custom_data_cols = set()
    register_columns('a')
    register_columns('b', 'b', 'c')
    register_columns(['d', 'e'], 'c')
    expected = {'a', 'b', 'c', 'd', 'e'}
    assert scope.custom_data_cols == expected
    assert scope.all_data_cols == scope.default_data_cols | expected

def register_columns(names: Union[str, Iterable[str]], *args):
    """Registers ``names`` of user-defined data columns."""
    StaticScope.instance().register_custom_cols(names, *args)

def test_register_columns_when_frozen_then_error(scope):
    scope.freeze_data_cols()
    with pytest.raises(ValueError, match=re.escape('Cannot modify columns when strategy is running.')):
        register_columns('a')
    scope.unfreeze_data_cols()

def test_unregister_columns(scope):
    scope.custom_data_cols = set()
    register_columns('a', 'b', 'c', 'd', 'e')
    unregister_columns('a', 'b')
    unregister_columns('c')
    unregister_columns(['c'], 'd')
    assert scope.custom_data_cols == {'e'}
    assert scope.all_data_cols == scope.default_data_cols | {'e'}

def unregister_columns(names: Union[str, Iterable[str]], *args):
    """Unregisters ``names`` of user-defined data columns."""
    StaticScope.instance().unregister_custom_cols(names, *args)

def test_unregister_columns_when_frozen_then_error(scope):
    scope.freeze_data_cols()
    with pytest.raises(ValueError, match=re.escape('Cannot modify columns when strategy is running.')):
        unregister_columns('a')
    scope.unfreeze_data_cols()

class TestColumnScope:

    def _assert_length(self, values, end_index, data_source_df, sym):
        df = data_source_df[data_source_df['symbol'] == sym]
        expected = df.shape[0] if end_index is None else end_index
        assert len(values) == expected

    def test_fetch_dict(self, col_scope, data_source_df, symbols, end_index):
        cols = ['date', 'close']
        result = col_scope.fetch_dict(symbols[0], cols, end_index)
        assert set(result.keys()) == set(cols)
        for value in result.values():
            self._assert_length(value, end_index, data_source_df, symbols[0])

    def test_fetch(self, col_scope, data_source_df, symbols, end_index):
        values = col_scope.fetch(symbols[0], 'close', end_index)
        assert isinstance(values, np.ndarray)
        self._assert_length(values, end_index, data_source_df, symbols[0])

    def test_fetch_when_cached(self, col_scope, data_source_df, symbols):
        col_scope.fetch(symbols[0], 'close', 1)
        values = col_scope.fetch(symbols[0], 'close', 2)
        assert isinstance(values, np.ndarray)
        self._assert_length(values, 2, data_source_df, symbols[0])

    def test_fetch_dict_when_empty_names(self, col_scope, symbols, end_index):
        result = col_scope.fetch_dict(symbols[0], [], end_index)
        assert not len(result)

    def test_fetch_dict_when_name_not_found(self, col_scope, symbols, end_index):
        result = col_scope.fetch_dict(symbols[0], ['foo'], end_index)
        assert result['foo'] is None

    def test_fetch_when_name_not_found(self, col_scope, symbols, end_index):
        assert col_scope.fetch(symbols[0], 'foo', end_index) is None

    def test_fetch_when_symbol_not_found_then_error(self, col_scope, end_index):
        with pytest.raises(ValueError, match=re.escape('Symbol not found: FOO.')):
            col_scope.fetch('FOO', 'close', end_index)

    def test_fetch_dict_when_symbol_not_found_then_error(self, col_scope, end_index):
        with pytest.raises(ValueError, match=re.escape('Symbol not found: FOO.')):
            col_scope.fetch_dict('FOO', ['close'], end_index)

    def test_fetch_dict_when_cached(self, col_scope, data_source_df, symbols, end_index):
        cols = ['date', 'close']
        col_scope.fetch_dict(symbols[0], cols, end_index)
        result = col_scope.fetch_dict(symbols[0], cols, end_index)
        assert set(result.keys()) == set(cols)
        for value in result.values():
            self._assert_length(value, end_index, data_source_df, symbols[0])

    def test_bar_data_from_data_columns(self, col_scope, data_source_df, symbols, end_index):
        register_columns('adj_close')
        bar_data = col_scope.bar_data_from_data_columns(symbols[0], end_index)
        sym_df = data_source_df[data_source_df['symbol'] == symbols[0]]
        for col in ('open', 'high', 'low', 'close', 'volume', 'adj_close'):
            assert (getattr(bar_data, col) == sym_df[col].to_numpy()[:end_index]).all()
        unregister_columns('adj_close')


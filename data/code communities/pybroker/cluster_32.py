# Cluster 32

class TestModelInputScope:

    def test_fetch(self, input_scope, model_source, symbol, data_source_df, end_index):
        df = data_source_df[data_source_df['symbol'] == symbol]
        result = input_scope.fetch(symbol, model_source.name, end_index)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == set(model_source.indicators)
        assert result.shape[0] == df.shape[0] if end_index is None else end_index

    def test_fetch_when_input_fn(self, scope, indicators, input_scope, symbol, data_source_df, end_index, trained_model):
        scope.custom_data_cols = set()
        expected_cols = {'hhv', 'llv', 'sumv'}

        def input_fn(df):
            assert set(df.columns) == expected_cols
            df['foo'] = np.ones(len(df['hhv']))
            return df
        model_source = model(trained_model.name, lambda *_: trained_model, indicators, input_data_fn=input_fn)
        df = data_source_df[data_source_df['symbol'] == symbol]
        result = input_scope.fetch(symbol, model_source.name, end_index)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {'foo'} | expected_cols
        assert result.shape[0] == df.shape[0] if end_index is None else end_index

    def test_fetch_when_cached(self, input_scope, model_source, symbol, data_source_df, end_index):
        input_scope.fetch(symbol, model_source.name, end_index)
        result = input_scope.fetch(symbol, model_source.name, end_index)
        df = data_source_df[data_source_df['symbol'] == symbol]
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == set(model_source.indicators)
        assert result.shape[0] == df.shape[0] if end_index is None else end_index

    @pytest.mark.parametrize('sym, name, expected_msg', [('FOO', MODEL_NAME, 'Symbol not found: FOO'), ('SPY', 'foo', "Model 'foo' not found.")])
    def test_fetch_when_not_found_then_error(self, input_scope, sym, name, expected_msg):
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            input_scope.fetch(sym, name)

def model(name: str, fn: Callable[..., Union[Any, tuple[Any, Iterable[str]]]], indicators: Optional[Iterable[Indicator]]=None, input_data_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]]=None, predict_fn: Optional[Callable[[Any, pd.DataFrame], NDArray]]=None, pretrained: bool=False, **kwargs) -> ModelSource:
    """Creates a :class:`.ModelSource` instance and registers it globally with
    ``name``.

    Args:
        name: Name for referencing the model globally.
        fn: :class:`Callable` used to either train or load a model instance. If
            for training, then ``fn`` has signature ``Callable[[symbol: str,
            train_data: DataFrame, test_data: DataFrame, ...], DataFrame]``.
            If for loading, then ``fn`` has signature
            ``Callable[[symbol: str, train_start_date: datetime,
            train_end_date: datetime, ...], DataFrame]``. This is expected to
            return either a trained model instance, or a tuple containing a
            trained model instance and a :class:`Iterable` of column names to
            to be used as input for the model when making predictions.
        indicators: :class:`Iterable` of
            :class:`pybroker.indicator.Indicator`\\ s used as features of the
            model.
        input_data_fn: :class:`Callable[[DataFrame], DataFrame]` for
            preprocessing input data passed to the model when making
            predictions. If set, ``input_data_fn`` will be called with a
            :class:`pandas.DataFrame` containing all test data.
        predict_fn: :class:`Callable[[Model, DataFrame], ndarray]` that
            overrides calling the model's default ``predict`` function. If set,
            ``predict_fn`` will be called with the trained model and a
            :class:`pandas.DataFrame` containing all test data.
        pretrained: If ``True``, then ``fn`` is used to load and return a
            pre-trained model. If ``False``, ``fn`` is used to train and return
            a new model. Defaults to ``False``.
        \\**kwargs: Additional arguments to pass to ``fn``.

    Returns:
        :class:`.ModelSource` instance.
    """
    scope = StaticScope.instance()
    indicator_names = tuple(sorted(set((ind.name for ind in indicators)))) if indicators is not None else tuple()
    if pretrained:
        loader = ModelLoader(name=name, load_fn=fn, indicator_names=indicator_names, input_data_fn=input_data_fn, predict_fn=predict_fn, kwargs=kwargs)
        scope.set_model_source(loader)
        return loader
    else:
        trainer = ModelTrainer(name=name, train_fn=fn, indicator_names=indicator_names, input_data_fn=input_data_fn, predict_fn=predict_fn, kwargs=kwargs)
        scope.set_model_source(trainer)
        return trainer

@pytest.fixture()
def exec_model_source(scope, data_source_df, indicators):
    return model(MODEL_NAME, lambda sym, *_: FakeModel(sym, np.full(data_source_df[data_source_df['symbol'] == sym].shape[0], 100)), indicators, pretrained=False)

@pytest.fixture(params=[True, False])
def model_source(scope, data_source_df, indicators, request):
    return model(MODEL_NAME, lambda sym, *_: FakeModel(sym, np.full(data_source_df[data_source_df['symbol'] == sym].shape[0], 100)), indicators, pretrained=request.param)

@pytest.fixture()
def model_loader():
    return model('loader', lambda symbol, train_start_date, train_end_date: FakeModel(symbol=symbol, preds=[]), [], pretrained=True)

@pytest.mark.parametrize('pretrained, input_cols', [(True, False), (True, False)])
def test_model(indicators, pretrained, input_cols):

    def input_data_fn(df):
        pass

    def predict_fn(model, df):
        pass
    name = f'pretrained={pretrained}'
    source = model(name, lambda x: (x, ['hhv', 'llv']) if input_cols else x, indicators, input_data_fn=input_data_fn, predict_fn=predict_fn, pretrained=pretrained)
    assert isinstance(source, ModelLoader if pretrained else ModelTrainer)
    assert source.name == name
    assert source.indicators == ('hhv', 'llv', 'sumv')
    assert source._input_data_fn is input_data_fn
    assert source._predict_fn is predict_fn


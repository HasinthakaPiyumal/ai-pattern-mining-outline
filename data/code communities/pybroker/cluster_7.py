# Cluster 7

class TestModelsMixin:

    def _assert_models(self, models, expected_model_syms):
        assert set(models.keys()) == set(expected_model_syms)
        for model_sym in expected_model_syms:
            model = models[model_sym]
            assert isinstance(model, TrainedModel)
            assert model.name == model_sym.model_name
            assert model.instance.symbol == model_sym.symbol

    @pytest.mark.usefixtures('setup_model_cache')
    @pytest.mark.parametrize('param_test_data', [pd.DataFrame(columns=['symbol', 'date']), LazyFixture('test_data')])
    def test_train_models(self, model_syms, train_data, param_test_data, ind_data, cache_date_fields, request):
        param_test_data = get_fixture(request, param_test_data)
        mixin = ModelsMixin()
        models = mixin.train_models(model_syms, train_data, param_test_data, ind_data, cache_date_fields)
        self._assert_models(models, model_syms)

    @pytest.mark.usefixtures('setup_model_cache')
    def test_train_models_when_empty_train_data(self, model_syms, test_data, ind_data, cache_date_fields):
        mixin = ModelsMixin()
        models = mixin.train_models(model_syms, pd.DataFrame(), test_data, ind_data, cache_date_fields)
        assert len(models) == 0

    @pytest.mark.usefixtures('setup_enabled_model_cache')
    def test_train_models_when_cached(self, model_syms, train_data, test_data, ind_data, cache_date_fields):
        mixin = ModelsMixin()
        mixin.train_models(model_syms, train_data, test_data, ind_data, cache_date_fields)
        models = mixin.train_models(model_syms, train_data, test_data, ind_data, cache_date_fields)
        self._assert_models(models, model_syms)

    @pytest.mark.usefixtures('setup_enabled_model_cache')
    def test_train_models_when_partial_cached(self, model_syms, train_data, test_data, ind_data, cache_date_fields):
        mixin = ModelsMixin()
        mixin.train_models(model_syms[:1], train_data, test_data, ind_data, cache_date_fields)
        models = mixin.train_models(model_syms, train_data, test_data, ind_data, cache_date_fields)
        self._assert_models(models, model_syms)

def get_fixture(request, param):
    if isinstance(param, LazyFixture):
        return request.getfixturevalue(param.name)
    return param


# Cluster 34

@pytest.fixture(params=[True, False])
def trained_models(model_source, preds, symbols, data_source_df, request):
    trained_models = {}
    for sym in symbols:
        model_sym = ModelSymbol(MODEL_NAME, sym)
        if request.param:

            def predict_fn(preds_array):

                def _(model, df):
                    return preds_array
                return _
            trained_models[model_sym] = TrainedModel(MODEL_NAME, FakeModel(sym, None), predict_fn=predict_fn(preds[sym]), input_cols=('hhv', 'llv', 'sumv'))
        else:
            trained_models[model_sym] = TrainedModel(MODEL_NAME, FakeModel(sym, preds[sym]), predict_fn=None, input_cols=('hhv', 'llv', 'sumv'))
    return trained_models

def predict_fn(preds_array):

    def _(model, df):
        return preds_array
    return _


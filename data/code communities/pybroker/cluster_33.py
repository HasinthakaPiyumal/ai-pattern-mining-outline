# Cluster 33

def test_get_signals(symbols, scope, col_scope, ind_scope, pred_scope, data_source_df, ind_data, preds):
    dfs = get_signals(symbols, col_scope, ind_scope, pred_scope)
    assert set(dfs.keys()) == set(symbols)
    for sym in symbols:
        for col in scope.all_data_cols:
            if col not in data_source_df.columns:
                continue
            assert np.array_equal(dfs[sym][col].values, data_source_df[data_source_df['symbol'] == sym][col].values)
        assert np.array_equal(dfs[sym][f'{MODEL_NAME}_pred'].values, preds[sym], equal_nan=True)
    for ind_name, sym in ind_data:
        assert np.array_equal(dfs[sym][ind_name].values, ind_data[IndicatorSymbol(ind_name, sym)].values, equal_nan=True)

def get_signals(symbols: Iterable[str], col_scope: ColumnScope, ind_scope: IndicatorScope, pred_scope: PredictionScope) -> dict[str, pd.DataFrame]:
    """Retrieves dictionary of :class:`pandas.DataFrame`\\ s
    containing bar data, indicator data, and model predictions for each symbol.
    """
    static_scope = StaticScope.instance()
    cols = static_scope.all_data_cols
    inds = static_scope._indicators.keys()
    models = static_scope._model_sources.keys()
    dates = col_scope._df.index.get_level_values(1)
    dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        data = {DataCol.DATE.value: dates}
        for col in cols:
            data[col] = col_scope.fetch(sym, col)
        for ind in inds:
            try:
                data[ind] = ind_scope.fetch(sym, ind)
            except ValueError:
                continue
        for model in models:
            try:
                data[f'{model}_pred'] = pred_scope.fetch(sym, model)
            except ValueError:
                continue
        dfs[sym] = pd.DataFrame(data)
    return dfs


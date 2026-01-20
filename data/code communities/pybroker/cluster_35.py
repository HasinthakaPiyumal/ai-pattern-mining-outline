# Cluster 35

def test_foreign_with_empty_col(scope, ctx, foreign, data_source_df, end_index):
    scope.register_custom_cols('adj_close')
    df = data_source_df[data_source_df['symbol'] == foreign]

    def verify_bar_data(bar_data):
        for col in ('date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'):
            assert (getattr(bar_data, col) == df[col].values[:end_index]).all()
    verify_bar_data(ctx.foreign(foreign))
    verify_bar_data(ctx.foreign(foreign))

def verify_bar_data(bar_data):
    for col in ('date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'):
        assert (getattr(bar_data, col) == df[col].values[:end_index]).all()


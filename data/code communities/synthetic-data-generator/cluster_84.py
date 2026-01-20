# Cluster 84

def test_chn_name_inspector_demo_data(raw_data):
    inspector_CHN_name = ChineseNameInspector()
    inspector_CHN_name.fit(raw_data)
    assert not inspector_CHN_name.regex_columns
    assert sorted(inspector_CHN_name.inspect()['chinese_name_columns']) == sorted([])
    assert inspector_CHN_name.inspect_level == 40
    assert inspector_CHN_name.pii is True

def test_chn_name_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_CHN_name = ChineseNameInspector()
    inspector_CHN_name.fit(chn_personal_test_df)
    assert inspector_CHN_name.regex_columns
    assert sorted(inspector_CHN_name.inspect()['chinese_name_columns']) == sorted(['chn_name'])
    assert inspector_CHN_name.inspect_level == 40
    assert inspector_CHN_name.pii is True


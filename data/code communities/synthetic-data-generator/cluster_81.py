# Cluster 81

def test_chn_postcode_inspector_demo_data(raw_data):
    inspector_PostCode = ChinaMainlandPostCode()
    inspector_PostCode.fit(raw_data)
    assert not inspector_PostCode.regex_columns
    assert sorted(inspector_PostCode.inspect()['china_mainland_postcode_columns']) == sorted([])
    assert inspector_PostCode.inspect_level == 20
    assert inspector_PostCode.pii is False

def test_chn_postcode_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_PostCode = ChinaMainlandPostCode()
    inspector_PostCode.fit(chn_personal_test_df)
    assert inspector_PostCode.regex_columns
    assert sorted(inspector_PostCode.inspect()['china_mainland_postcode_columns']) == sorted(['postcode'])
    assert inspector_PostCode.inspect_level == 20
    assert inspector_PostCode.pii is False


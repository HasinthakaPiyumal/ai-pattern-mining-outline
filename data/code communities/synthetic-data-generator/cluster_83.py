# Cluster 83

def test_chn_address_inspector_demo_data(raw_data):
    inspector_CHN_Address = ChinaMainlandAddressInspector()
    inspector_CHN_Address.fit(raw_data)
    assert not inspector_CHN_Address.regex_columns
    assert sorted(inspector_CHN_Address.inspect()['china_mainland_address_columns']) == sorted([])
    assert inspector_CHN_Address.inspect_level == 30
    assert inspector_CHN_Address.pii is True

def test_chn_address_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_CHN_Address = ChinaMainlandAddressInspector()
    inspector_CHN_Address.fit(chn_personal_test_df)
    assert inspector_CHN_Address.regex_columns
    assert sorted(inspector_CHN_Address.inspect()['china_mainland_address_columns']) == sorted(['chn_address'])
    assert inspector_CHN_Address.inspect_level == 30
    assert inspector_CHN_Address.pii is True


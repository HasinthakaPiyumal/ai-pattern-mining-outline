# Cluster 79

def test_chn_phone_inspector_demo_data(raw_data):
    inspector_Phone = ChinaMainlandMobilePhoneInspector()
    inspector_Phone.fit(raw_data)
    assert not inspector_Phone.regex_columns
    assert sorted(inspector_Phone.inspect()['china_mainland_mobile_phone_columns']) == sorted([])
    assert inspector_Phone.inspect_level == 30
    assert inspector_Phone.pii is True

def test_chn_phone_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_Phone = ChinaMainlandMobilePhoneInspector()
    inspector_Phone.fit(chn_personal_test_df)
    assert inspector_Phone.regex_columns
    assert sorted(inspector_Phone.inspect()['china_mainland_mobile_phone_columns']) == sorted(['mobile_phone_no'])
    assert inspector_Phone.inspect_level == 30
    assert inspector_Phone.pii is True


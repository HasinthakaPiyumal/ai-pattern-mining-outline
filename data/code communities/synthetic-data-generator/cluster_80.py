# Cluster 80

def test_chn_ID_inspector_demo_data(raw_data):
    inspector_ID = ChinaMainlandIDInspector()
    inspector_ID.fit(raw_data)
    assert not inspector_ID.regex_columns
    assert sorted(inspector_ID.inspect()['china_mainland_id_columns']) == sorted([])
    assert inspector_ID.inspect_level == 30
    assert inspector_ID.pii is True

def test_chn_ID_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_ID = ChinaMainlandIDInspector()
    inspector_ID.fit(chn_personal_test_df)
    assert inspector_ID.regex_columns
    assert sorted(inspector_ID.inspect()['china_mainland_id_columns']) == sorted(['ssn_sfz'])
    assert inspector_ID.inspect_level == 30
    assert inspector_ID.pii is True


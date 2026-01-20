# Cluster 85

def test_eng_name_inspector_demo_data(raw_data):
    inspector_ENG_name = EnglishNameInspector()
    inspector_ENG_name.fit(raw_data)
    assert not inspector_ENG_name.regex_columns
    assert sorted(inspector_ENG_name.inspect()['english_name_columns']) == sorted([])
    assert inspector_ENG_name.inspect_level == 40
    assert inspector_ENG_name.pii is True

def test_eng_name_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_ENG_name = EnglishNameInspector()
    inspector_ENG_name.fit(chn_personal_test_df)
    assert inspector_ENG_name.regex_columns
    assert sorted(inspector_ENG_name.inspect()['english_name_columns']) == sorted(['eng_name'])
    assert inspector_ENG_name.inspect_level == 40
    assert inspector_ENG_name.pii is True


# Cluster 86

def test_chn_company_inspector_demo_data(raw_data):
    inspector_PostCode = ChineseCompanyNameInspector()
    inspector_PostCode.fit(raw_data)
    assert not inspector_PostCode.regex_columns
    assert sorted(inspector_PostCode.inspect()['chinese_company_name_columns']) == sorted([])
    assert inspector_PostCode.inspect_level == 40
    assert inspector_PostCode.pii is False

def test_chn_company_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_PostCode = ChineseCompanyNameInspector()
    inspector_PostCode.fit(chn_personal_test_df)
    assert inspector_PostCode.regex_columns
    assert sorted(inspector_PostCode.inspect()['chinese_company_name_columns']) == sorted(['company_name'])
    assert inspector_PostCode.inspect_level == 40
    assert inspector_PostCode.pii is False


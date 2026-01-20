# Cluster 82

def test_chn_uscc_inspector_demo_data(raw_data):
    inspector_USCC = ChinaMainlandUnifiedSocialCreditCode()
    inspector_USCC.fit(raw_data)
    assert not inspector_USCC.regex_columns
    assert sorted(inspector_USCC.inspect()['unified_social_credit_code_columns']) == sorted([])
    assert inspector_USCC.inspect_level == 30
    assert inspector_USCC.pii is True

def test_chn_uscc_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_USCC = ChinaMainlandUnifiedSocialCreditCode()
    inspector_USCC.fit(chn_personal_test_df)
    assert inspector_USCC.regex_columns
    assert sorted(inspector_USCC.inspect()['unified_social_credit_code_columns']) == sorted(['uscc'])
    assert inspector_USCC.inspect_level == 30
    assert inspector_USCC.pii is True


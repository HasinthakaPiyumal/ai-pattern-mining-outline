# Cluster 78

def test_email_inspector_demo_data(raw_data):
    inspector_Email = EmailInspector()
    inspector_Email.fit(raw_data)
    assert not inspector_Email.regex_columns
    assert sorted(inspector_Email.inspect()['email_columns']) == sorted([])
    assert inspector_Email.inspect_level == 30
    assert inspector_Email.pii is True

def test_email_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_Email = EmailInspector()
    inspector_Email.fit(chn_personal_test_df)
    assert inspector_Email.regex_columns
    assert sorted(inspector_Email.inspect()['email_columns']) == sorted(['email'])
    assert inspector_Email.inspect_level == 30
    assert inspector_Email.pii is True


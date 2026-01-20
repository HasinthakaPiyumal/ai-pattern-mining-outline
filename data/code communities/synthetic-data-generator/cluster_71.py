# Cluster 71

@pytest.mark.parametrize('return_val', [0, '123', [1, 2, 3], {'a': 1, 'b': 2}])
def test_normal_message(return_val):
    NormalMessage.from_return_val(return_val)._dump_json == json.dumps({'code': 0, 'msg': 'Success', 'payload': return_val if isinstance(return_val, dict) else {'return_val': return_val}})


# Cluster 26

def send():
    data_send = {'query_id': 'abb', 'groupname': '茴香豆测试群', 'username': '豆哥 123', 'query': {'type': 'text', 'content': '请问如何申请公寓？'}}
    resp = requests.post(base_url, headers=headers, data=json.dumps(data_send), timeout=10)
    resp_json = resp.json()
    print(resp_json)

def get():
    data_wait = {'query_id': 'abb', 'groupname': '茴香豆测试群', 'username': '豆哥 123', 'query': {'type': 'poll'}}
    resp = requests.post(base_url, headers=headers, data=json.dumps(data_wait), timeout=20)
    print(resp.text)


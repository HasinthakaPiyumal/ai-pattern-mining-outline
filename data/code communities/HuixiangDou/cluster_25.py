# Cluster 25

def test_chat():
    queries = ['请问公寓退房需要注意哪些事情？']
    for query in queries:
        target = {'type': TaskCode.CHAT.value, 'payload': {'query_id': 'ae86', 'feature_store_id': '9527', 'content': query, 'images': [], 'history': [{'sender': 0, 'content': '你好'}, {'sender': 0, 'content': '你是谁'}, {'sender': 1, 'content': '我是行政助手茴香豆'}]}}
        task_in.put(json.dumps(target, ensure_ascii=False))
        print(chat_out.get())


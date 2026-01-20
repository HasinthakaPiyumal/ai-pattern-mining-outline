# Cluster 56

class User:

    def __init__(self):
        self.history = []
        self.last_msg_time = time.time()
        self.last_msg_id = -1
        self.last_process_time = -1
        self._id = ''
        self.group_id = ''

    def __str__(self):
        obj = {'history': [], 'last_msg_time': self.last_msg_time, 'last_process_time': self.last_process_time, '_id': self._id}
        for item in self.history:
            obj['history'].append(convert_talk_to_dict(item))
        return json.dumps(obj, indent=2, ensure_ascii=False)

    def feed(self, msg: Message):
        if msg.type in ['url', 'image']:
            talk = Talk(query=msg.query, refs=msg.url)
            self.history.append(talk)
        else:
            talk = Talk(query=msg.query)
            self.history.append(talk)
        self.last_msg_time = time.time()
        self.last_msg_type = msg.type
        self.last_msg_id = msg._id
        self._id = msg.global_user_id
        self.group_id = msg.group_id

    def concat(self):
        if len(self.history) < 2:
            return
        ret = []
        merge_list = []
        now = time.time()
        for item in self.history:
            if abs(now - item.now) > 7200:
                continue
            answer = item.reply
            if answer is not None and len(answer) > 0:
                ret.append(item)
            else:
                merge_list.append(item.query)
        concat_query = '\n'.join(merge_list)
        concat_talk = Talk(query=concat_query)
        ret.append(concat_talk)
        self.history = ret

    def update_history(self, query, reply, refs):
        if type(refs) is list:
            talk = Talk(query=query, reply=reply, refs=tuple(refs))
        else:
            talk = Talk(query=query, reply=reply, refs=refs)
        self.history[-1] = talk
        self.last_process_time = time.time()

def convert_talk_to_dict(talk: Talk):
    return {'query': talk.query, 'reply': talk.reply, 'refs': talk.refs, 'now': talk.now}


# Cluster 55

def do_p2_im_message_receive_v1(data: P2ImMessageReceiveV1) -> None:
    logger.info(lark.JSON.marshal(data))
    if data.header.event_type != 'im.message.receive_v1':
        return None
    msg = data.event.message
    if msg.chat_type != 'group':
        return None
    if msg.message_type != 'text':
        return None
    msg_id = msg.message_id
    group_id = msg.chat_id
    user_id = data.event.sender.sender_id.user_id
    content = json.loads(msg.content)['text']
    msg_time = msg.create_time
    json_str = json.dumps({'source': 'lark', 'msg_id': msg_id, 'user_id': user_id, 'content': content, 'group_id': group_id, 'msg_time': msg_time}, ensure_ascii=False) + '\n'
    que = None
    if is_revert_command(content):
        que = Queue(name='huixiangdou-high-priority')
    else:
        que = Queue(name='huixiangdou')
    que.put(json_str)
    logger.debug(f'save {json_str} to {que.key}')
    return None

def is_revert_command(content: str):
    if '豆哥撤回' in content:
        return True
    return False

class WkteamManager:
    """
    1. wkteam Login, see https://wkteam.cn/
    2. Handle wkteam wechat message call back
    """

    def __init__(self, config_path: str):
        """init with config."""
        self.WKTEAM_IP_PORT = '121.229.29.88:9899'
        self.auth = ''
        self.wId = ''
        self.wcId = ''
        self.qrCodeUrl = ''
        self.wkteam_config = dict()
        self.users = dict()
        self.preprocessed = set()
        self.messages = []
        self.group_whitelist = dict()
        with open(config_path, encoding='utf8') as f:
            self.config = pytoml.load(f)
            assert len(self.config) > 1
            wkconf = self.config['frontend']['wechat_wkteam']
            for key in wkconf.keys():
                key = key.strip()
                if re.match('\\d+', key) is None:
                    continue
                group = wkconf[key]
                self.group_whitelist['{}@chatroom'.format(key)] = group['name']
            self.wkteam_config = types.SimpleNamespace(**wkconf)
        if os.getenv('REDIS_HOST') is None:
            os.environ['REDIS_HOST'] = str(self.wkteam_config.redis_host)
        if os.getenv('REDIS_PORT') is None:
            os.environ['REDIS_PORT'] = str(self.wkteam_config.redis_port)
        if os.getenv('REDIS_PASSWORD') is None:
            os.environ['REDIS_PASSWORD'] = str(self.wkteam_config.redis_passwd)
        if not os.path.exists(self.wkteam_config.dir):
            os.makedirs(self.wkteam_config.dir)
        self.license_path = os.path.join(self.wkteam_config.dir, 'license.json')
        self.record_path = os.path.join(self.wkteam_config.dir, 'record.jsonl')
        if os.path.exists(self.license_path):
            with open(self.license_path) as f:
                jsonobj = json.load(f)
                self.auth = jsonobj['auth']
                self.wId = jsonobj['wId']
                self.wcId = jsonobj['wcId']
                self.qrCodeUrl = jsonobj['qrCodeUrl']
                logger.debug(jsonobj)
        self.sent_msg = dict()
        self.debug()

    def debug(self):
        logger.debug('auth {}'.format(self.auth))
        logger.debug('wId {}'.format(self.wId))
        logger.debug('wcId {}'.format(self.wcId))
        logger.debug('REDIS_HOST {}'.format(os.getenv('REDIS_HOST')))
        logger.debug('REDIS_PORT {}'.format(os.getenv('REDIS_PORT')))
        logger.debug(self.group_whitelist)

    def post(self, url, data, headers):
        """Wrap http post and error handling."""
        resp = requests.post(url, data=json.dumps(data), headers=headers)
        json_str = resp.content.decode('utf8')
        logger.debug(json_str)
        if resp.status_code != 200:
            return (None, Exception('wkteam auth fail {}'.format(json_str)))
        json_obj = json.loads(json_str)
        if json_obj['code'] != '1000':
            return (json_obj, Exception(json_str))
        return (json_obj, None)

    def revert_all(self):
        for groupId in self.group_whitelist:
            self.revert(groupId=groupId)

    def revert(self, groupId: str):
        """Revert all msgs in this group."""
        if groupId in self.group_whitelist:
            groupname = self.group_whitelist[groupId]
            logger.debug('revert message in group {} {}'.format(groupname, groupId))
        else:
            logger.debug('revert message in group {} '.format(groupId))
        if groupId not in self.sent_msg:
            return
        group_sent_list = self.sent_msg[groupId]
        for sent in group_sent_list:
            logger.info(sent)
            time_diff = abs(time.time() - int(sent['createTime']))
            if time_diff <= 120:
                headers = {'Content-Type': 'application/json', 'Authorization': self.auth}
                self.post(url='http://{}/revokeMsg'.format(self.WKTEAM_IP_PORT), data=sent, headers=headers)
        del self.sent_msg[groupId]

    def download_image(self, param: dict):
        """Download group chat image."""
        content = param['content']
        msgId = param['msgId']
        wId = param['wId']
        if len(self.auth) < 1:
            logger.error('Authentication empty')
            return
        headers = {'Content-Type': 'application/json', 'Authorization': self.auth}
        data = {'wId': wId, 'content': content, 'msgId': msgId, 'type': 0}

        def generate_hash_filename(data: dict):
            xstr = json.dumps(data)
            md5 = hashlib.md5()
            md5.update(xstr.encode('utf8'))
            return md5.hexdigest()[0:6] + '.jpg'

        def download(data: dict, headers: dict, dir: str):
            resp = requests.post('http://{}/getMsgImg'.format(self.WKTEAM_IP_PORT), data=json.dumps(data), headers=headers)
            json_str = resp.content.decode('utf8')
            if resp.status_code == 200:
                jsonobj = json.loads(json_str)
                if jsonobj['code'] != '1000':
                    logger.error('download {} {}'.format(data, json_str))
                    return
                image_url = jsonobj['data']['url']
                logger.info('image url {}'.format(image_url))
                resp = requests.get(image_url, stream=True)
                image_path = None
                if resp.status_code == 200:
                    image_dir = os.path.join(dir, 'images')
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                    image_path = os.path.join(image_dir, generate_hash_filename(data=data))
                    logger.debug('local path {}'.format(image_path))
                    with open(image_path, 'wb') as image_file:
                        for chunk in resp.iter_content(1024):
                            image_file.write(chunk)
                return (image_url, image_path)
        url = ''
        path = ''
        try:
            url, path = download(data, headers, self.wkteam_config.dir)
        except Exception as e:
            logger.error(str(e))
            return (None, None)
        return (url, path)

    def login(self):
        """user login, need scan qr code on mobile phone."""
        if len(self.wkteam_config.account) < 1 or len(self.wkteam_config.password) < 1:
            return Exception('wkteam account or password not set')
        if len(self.wkteam_config.callback_ip) < 1:
            return Exception('wkteam wechat message public callback ip not set, try FRP or buy cloud service ?')
        if self.wkteam_config.proxy <= 0:
            return Exception('wkteam proxy not set')
        headers = {'Content-Type': 'application/json'}
        data = {'account': self.wkteam_config.account, 'password': self.wkteam_config.password}
        json_obj, err = self.post(url='http://{}/member/login'.format(self.WKTEAM_IP_PORT), data=data, headers=headers)
        if err is not None:
            return err
        self.auth = json_obj['data']['Authorization']
        headers['Authorization'] = self.auth
        data = {'wcId': '', 'proxy': self.wkteam_config.proxy}
        json_obj, err = self.post(url='http://{}/iPadLogin'.format(self.WKTEAM_IP_PORT), data=data, headers=headers)
        if err is not None:
            return err
        x = json_obj['data']
        self.wId = x['wId']
        self.qrCodeUrl = x['qrCodeUrl']
        logger.info('浏览器打开这个地址、下载二维码。打开手机，扫描登录微信\n {}\n 请确认 proxy 地区正确，首次使用、24 小时后要再次登录，以后不需要登。'.format(self.qrCodeUrl))
        json_obj, err = self.post(url='http://{}/getIPadLoginInfo'.format(self.WKTEAM_IP_PORT), data={'wId': self.wId}, headers=headers)
        x = json_obj['data']
        self.wcId = x['wcId']
        with open(self.license_path, 'w') as f:
            json_str = json.dumps({'auth': self.auth, 'wId': self.wId, 'wcId': self.wcId, 'qrCodeUrl': self.qrCodeUrl}, indent=2, ensure_ascii=False)
            f.write(json_str)

    def set_callback(self):
        httpUrl = 'http://{}:{}/callback'.format(self.wkteam_config.callback_ip, self.wkteam_config.callback_port)
        logger.debug('set callback url {}'.format(httpUrl))
        headers = {'Content-Type': 'application/json', 'Authorization': self.auth}
        data = {'httpUrl': httpUrl, 'type': 2}
        json_obj, err = self.post(url='http://{}/setHttpCallbackUrl'.format(self.WKTEAM_IP_PORT), data=data, headers=headers)
        if err is not None:
            return err
        logger.info('login success, all license saved to {}'.format(self.license_path))
        return None

    def send_image(self, groupId: str, image_url: str):
        headers = {'Content-Type': 'application/json', 'Authorization': self.auth}
        data = {'wId': self.wId, 'wcId': groupId, 'content': image_url}
        json_obj, err = self.post(url='http://{}/sendImage2'.format(self.WKTEAM_IP_PORT), data=data, headers=headers)
        if err is not None:
            return err
        sent = json_obj['data']
        sent['wId'] = self.wId
        if groupId not in self.sent_msg:
            self.sent_msg[groupId] = [sent]
        else:
            self.sent_msg[groupId].append(sent)
        return None

    def send_emoji(self, groupId: str, md5: str, length: int):
        headers = {'Content-Type': 'application/json', 'Authorization': self.auth}
        data = {'wId': self.wId, 'wcId': groupId, 'imageMd5': md5, 'imgSize': length}
        json_obj, err = self.post(url='http://{}/sendEmoji'.format(self.WKTEAM_IP_PORT), data=data, headers=headers)
        if err is not None:
            return err
        sent = json_obj['data']
        sent['wId'] = self.wId
        if groupId not in self.sent_msg:
            self.sent_msg[groupId] = [sent]
        else:
            self.sent_msg[groupId].append(sent)
        return None

    def send_message(self, groupId: str, text: str):
        headers = {'Content-Type': 'application/json', 'Authorization': self.auth}
        data = {'wId': self.wId, 'wcId': groupId, 'content': text}
        json_obj, err = self.post(url='http://{}/sendText'.format(self.WKTEAM_IP_PORT), data=data, headers=headers)
        if err is not None:
            return err
        sent = json_obj['data']
        sent['wId'] = self.wId
        if groupId not in self.sent_msg:
            self.sent_msg[groupId] = [sent]
        else:
            self.sent_msg[groupId].append(sent)
        return None

    def send_user_message(self, userId: str, text: str):
        headers = {'Content-Type': 'application/json', 'Authorization': self.auth}
        data = {'wId': self.wId, 'wcId': userId, 'content': text}
        json_obj, err = self.post(url='http://{}/sendText'.format(self.WKTEAM_IP_PORT), data=data, headers=headers)
        if err is not None:
            return err
        sent = json_obj['data']
        sent['wId'] = self.wId
        if userId not in self.sent_msg:
            self.sent_msg[userId] = [sent]
        else:
            self.sent_msg[userId].append(sent)
        return None

    def send_url(self, groupId: str, description: str, title: str, thumb_url: str, url: str):
        headers = {'Content-Type': 'application/json', 'Authorization': self.auth}
        data = {'wId': self.wId, 'wcId': groupId, 'description': description, 'title': title, 'thumbUrl': thumb_url, 'url': url}
        json_obj, err = self.post(url='http://{}/sendUrl'.format(self.WKTEAM_IP_PORT), data=data, headers=headers)
        if err is not None:
            return err
        sent = json_obj['data']
        sent['wId'] = self.wId
        if groupId not in self.sent_msg:
            self.sent_msg[groupId] = [sent]
        else:
            self.sent_msg[groupId].append(sent)
        return None

    def bind(self, logdir: str, port: int, forward: bool=False):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        logpath = os.path.join(logdir, 'wechat_message.jsonl')

        async def forward_msg(input_json: dict):
            msg = Message()
            print(input_json)
            err = msg.parse(wx_msg=input_json, bot_wxid=self.wId, auth=self.auth, wkteam_ip_port=self.WKTEAM_IP_PORT)
            if err is not None:
                logger.error(str(err))
                return
            if msg.new_msg_id in self.preprocessed:
                print(f'{msg.new_msg_id} repeated, skip')
                return
            self.preprocessed.add(msg.new_msg_id)
            come_from_whitelist = False
            from_group_name = ''
            for groupId, groupname in self.group_whitelist.items():
                if msg.group_id == groupId:
                    come_from_whitelist = True
                    from_group_name = groupname
            if not come_from_whitelist:
                return
            if msg.sender == self.wcId:
                return
            for groupId, _ in self.group_whitelist.items():
                if groupId == msg.group_id:
                    continue
                logger.info(str(msg.__dict__))
                if msg.type == 'text':
                    username = msg.push_content.split(':')[0].strip()
                    formatted_reply = '{}：{}'.format(username, msg.content)
                    self.send_message(groupId=groupId, text=formatted_reply)
                elif msg.type == 'image':
                    self.send_image(groupId=groupId, image_url=msg.url)
                elif msg.type == 'emoji':
                    self.send_emoji(groupId=groupId, md5=msg.md5, length=msg.length)
                elif msg.type == 'ref_for_others' or msg.type == 'ref_for_bot':
                    formatted_reply = '{}\n---\n{}'.format(msg.content, msg.query)
                    self.send_message(groupId=groupId, text=formatted_reply)
                elif msg.type == 'link':
                    thumbnail = msg.thumb_url if msg.thumb_url else 'https://deploee.oss-cn-shanghai.aliyuncs.com/icon.jpg'
                    self.send_url(groupId=groupId, description=msg.desc, title=msg.title, thumb_url=thumbnail, url=msg.url)
                await asyncio.sleep(random.uniform(0.2, 2.0))

        async def msg_callback(request):
            """Save wechat message to redis, for revert command, use high
            priority."""
            input_json = await request.json()
            with open(logpath, 'a') as f:
                json_str = json.dumps(input_json, indent=2, ensure_ascii=False)
                f.write(json_str)
                f.write('\n')
            logger.debug(input_json)
            try:
                msg_que = Queue(name='wechat')
            except Exception as e:
                msg_que = None
                print('redis unavailable')
                pass
            if input_json['messageType'] == '00000':
                return web.json_response(text='done')
            try:
                json_str = json.dumps(input_json)
                if is_revert_command(input_json):
                    self.revert_all()
                    return web.json_response(text='done')
                if msg_que:
                    msg_que.put(json_str)
                if forward and (not is_revert_command(input_json)):
                    await forward_msg(input_json)
            except Exception as e:
                logger.error(str(e))
            return web.json_response(text='done')
        app = web.Application()
        app.add_routes([web.post('/callback', msg_callback)])
        web.run_app(app, host='0.0.0.0', port=port)

    def serve(self, forward: bool=False):
        p = Process(target=self.bind, args=(self.wkteam_config.dir, self.wkteam_config.callback_port, forward))
        p.start()
        self.set_callback()
        p.join()

    def fetch_groupchats(self, user: User, max_length: int=12):
        """Before obtaining user messages, there are a maximum of `max_length`
        historical conversations in the group.

        Fetch them for coreference resolution.
        """
        user_msg_id = user.last_msg_id
        conversations = []
        for index in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[index]
            if len(conversations) >= max_length:
                break
            if msg.type == 'unknown':
                continue
            if msg._id < user_msg_id and msg.group_id == user.group_id:
                conversations.append(msg)
        return conversations

    async def loop(self, assistant):
        """Fetch all messages from redis, split it by groupId; concat by
        timestamp."""
        from huixiangdou.services import ErrorCode, kimi_ocr
        que = Queue(name='wechat')
        while True:
            for wx_msg_str in que.get_all():
                wx_msg = json.loads(wx_msg_str)
                logger.debug(wx_msg)
                msg = Message()
                err = msg.parse(wx_msg=wx_msg, bot_wxid=self.wcId, auth=self.auth, wkteam_ip_port=self.WKTEAM_IP_PORT)
                if err is not None:
                    logger.debug(str(err))
                    continue
                if msg.type == 'image':
                    _, local_image_path = self.download_image(param=msg.data)
                    llm_server_config = self.config['llm']['server']
                    if local_image_path is not None and llm_server_config['remote_type'] == 'kimi':
                        token = llm_server_config['remote_api_key']
                        msg.query = kimi_ocr(local_image_path, token)
                        logger.debug('kimi ocr {} {}'.format(local_image_path, msg.query))
                if len(msg.query) < 1:
                    continue
                self.messages.append(msg)
                if msg.type == 'ref_for_others':
                    continue
                if msg.global_user_id not in self.users:
                    self.users[msg.global_user_id] = User()
                user = self.users[msg.global_user_id]
                user.feed(msg)
            for user in self.users.values():
                if len(user.history) < 1:
                    continue
                now = time.time()
                if now - user.last_msg_time >= 12 and user.last_process_time < user.last_msg_time:
                    if user.last_msg_type in ['link', 'image']:
                        continue
                    logger.debug('before concat {}'.format(user))
                    user.concat()
                    logger.debug('after concat {}'.format(user))
                    assert len(user.history) > 0
                    item = user.history[-1]
                    if item.reply is not None and len(item.reply) > 0:
                        logger.error('item reply not None, {}'.format(item))
                    query = item.query
                    code = ErrorCode.QUESTION_TOO_SHORT
                    resp = ''
                    refs = []
                    groupname = ''
                    groupchats = []
                    if user.group_id in self.group_whitelist:
                        groupname = self.group_whitelist[user.group_id]
                    if len(query) >= 8:
                        groupchats = self.fetch_groupchats(user=user)
                        tuple_history = convert_history_to_tuple(user.history[0:-1])
                        async for sess in assistant.generate(query=query, history=tuple_history, groupname=groupname, groupchats=groupchats):
                            code, resp, refs = (sess.code, sess.response, sess.references)
                    user.last_process_time = time.time()
                    if code in [ErrorCode.NOT_A_QUESTION, ErrorCode.SECURITY, ErrorCode.NO_SEARCH_RESULT, ErrorCode.NO_TOPIC]:
                        del user.history[-1]
                    else:
                        user.update_history(query=query, reply=resp, refs=refs)
                    if code == ErrorCode.SUCCESS:
                        formatted_reply = ''
                        if len(query) > 30:
                            formatted_reply = '{}..\n---\n{}'.format(query[0:30], resp)
                        else:
                            formatted_reply = '{}\n---\n{}'.format(query, resp)
                        if user.group_id in self.group_whitelist:
                            logger.warning('send {} to {}'.format(formatted_reply, user.group_id))
                            self.send_message(groupId=user.group_id, text=formatted_reply)
                        else:
                            logger.warning('prepare respond {} to {}'.format(formatted_reply, user.group_id))


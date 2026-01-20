# Cluster 41

def get_wechat_on_message_url(suffix: str) -> str:
    endpoint = HuixiangDouEnv.get_message_endpoint()
    return endpoint + 'api/v1/message/v1/wechat/' + suffix

def get_lark_on_message_url() -> str:
    endpoint = HuixiangDouEnv.get_message_endpoint()
    return endpoint + 'api/v1/message/v1/lark'

def get_store_dir(feature_store_id: str) -> Union[str, None]:
    if not feature_store_id:
        return None
    try:
        crt_path = os.path.abspath(__file__)
        parent_path = os.path.dirname(os.path.dirname(crt_path))
        qa_path = os.path.join(parent_path, f'qa/{feature_store_id}')
        os.makedirs(qa_path, exist_ok=True)
        return qa_path
    except Exception as e:
        logger.error(f'{e}')
        return None

class QaLibCache:

    def __init__(self):
        pass

    @classmethod
    def get_qalib_info(cls, feature_store_id: str) -> Union[QalibInfo, None]:
        name = biz_const.RDS_KEY_QALIB_INFO
        key = feature_store_id
        o = r.hget(name, key)
        if not o:
            logger.error(f'[qalib] feature_store_id: {feature_store_id}, get info empty')
            return None
        return QalibInfo(**json.loads(o))

    @classmethod
    def set_qalib_info(cls, feature_store_id: str, info: QalibInfo):
        name = biz_const.RDS_KEY_QALIB_INFO
        key = feature_store_id
        return r.hset(name, key, info.model_dump_json()) == 1

    @classmethod
    def init_qalib_info(cls, feature_store_id: str, status: int, name: str, suffix: str) -> bool:
        """add qalib info to qalib:info db.

        :param name:
        :param feature_store_id:
        :param status:
        :return:
        """
        wechat = Wechat(onMessageUrl=get_wechat_on_message_url(suffix))
        lark = Lark(encryptKey=HuixiangDouEnv.get_lark_encrypt_key(), verificationToken=HuixiangDouEnv.get_lark_verification_token(), eventUrl=get_lark_on_message_url())
        qalib_info = QalibInfo(featureStoreId=feature_store_id, status=status, name=name, wechat=wechat, lark=lark, suffix=suffix)
        if not cls.set_qalib_info(feature_store_id, qalib_info):
            logger.error(f'[qalib] feature_store_id: {feature_store_id}, init qalib info failed')
            r.hdel(biz_const.RDS_KEY_QALIB_INFO, feature_store_id)
            return False
        return True

    @classmethod
    def del_qalib_info(cls, feature_store_id: str) -> bool:
        """del qalib info to qalib:info db.

        :param feature_store_id:
        :return:
        """
        return True if r.hdel(biz_const.RDS_KEY_QALIB_INFO, feature_store_id) == 1 else False

    @classmethod
    def rewrite_qalib_docs(cls, feature_store_id: str, added_docs: List[str], file_base: str) -> bool:
        """update qalib's docs.

        :param feature_store_id:
        :param added_docs:
        :param file_base:
        :return:
        """
        try:
            info = cls.get_qalib_info(feature_store_id)
            if not info:
                return False
            info.docs = list(set(added_docs))
            info.docBase = file_base
            cls.set_qalib_info(feature_store_id, info)
            return True
        except Exception as e:
            logger.error(f'[qalib] feature_store_id: {feature_store_id}, update docs failed: {e}')
            return False

    @classmethod
    def update_qalib_docs(cls, feature_store_id: str, added_docs: List[str], file_base: str) -> bool:
        """update qalib's docs.

        :param feature_store_id:
        :param added_docs:
        :param file_base:
        :return:
        """
        try:
            info = cls.get_qalib_info(feature_store_id)
            if not info:
                return False
            raw_docs = info.docs
            if not raw_docs:
                raw_docs = []
            raw_docs.extend(added_docs)
            info.docs = list(set(raw_docs))
            info.docBase = file_base
            cls.set_qalib_info(feature_store_id, info)
            return True
        except Exception as e:
            logger.error(f'[qalib] feature_store_id: {feature_store_id}, update docs failed: {e}')
            return False

    @classmethod
    def get_sample_info(cls, feature_store_id: str) -> Union[QalibSample, None]:
        o = r.hget(name=biz_const.RDS_KEY_SAMPLE_INFO, key=feature_store_id)
        if not o:
            logger.info(f'[qalib] feature_store_id: {feature_store_id}, get empty sample')
            return None
        return QalibSample(**json.loads(o))

    @classmethod
    def set_sample_info(cls, feature_store_id: str, sample_info: QalibSample):
        r.hset(name=biz_const.RDS_KEY_SAMPLE_INFO, key=feature_store_id, value=sample_info.model_dump_json())

    @classmethod
    def set_suffix_to_qalib(cls, suffix: str, feature_store_id: str):
        r.hset(name=biz_const.RDS_KEY_SUFFIX_TO_QALIB, key=suffix, value=feature_store_id)

    @classmethod
    def get_qalib_feature_store_id_by_suffix(cls, suffix: str) -> Union[str, None]:
        o = r.hget(name=biz_const.RDS_KEY_SUFFIX_TO_QALIB, key=suffix)
        if not o:
            logger.error(f'[qalib] suffix: {suffix} has no qalib')
            return None
        return o.decode('utf-8')

    @classmethod
    def get_lark_info_by_app_id(cls, app_id: str) -> Union[str, None]:
        key = biz_const.RDS_KEY_LARK_CONFIG + ':' + app_id
        o = r.get(key)
        if not o:
            logger.error(f'f[lark] app_id: {app_id} has no record')
            return None
        return o.decode('utf-8')

    @classmethod
    def set_lark_info(cls, app_id: str, app_secret: str):
        key = biz_const.RDS_KEY_LARK_CONFIG + ':' + app_id
        r.set(key, app_secret)

class ChatService:

    def __init__(self, request: Request, response: Response, hxd_info: QalibInfo):
        self.hxd_info = hxd_info
        self.request = request
        self.response = response

    async def chat_online(self, body: ChatRequestBody):
        feature_store_id = self.hxd_info.featureStoreId
        query_id = self.generate_query_id(body.content)
        logger.info(f'[chat-request]/online feature_store_id: {feature_store_id}, content: {body.content}, query_id: {query_id}')
        images_path = []
        if len(body.images) > 0:
            images_path = self._store_images(body.images, query_id)
            if len(images_path) == 0:
                return standard_error_response(biz_constant.ERR_CHAT)
        task = HxdTask(type=HxdTaskType.CHAT, payload=HxdTaskPayload(feature_store_id=feature_store_id, query_id=query_id, content=body.content, history=body.history, images=images_path))
        if HuixiangDouTask().updateTask(task):
            chat_query_info = ChatQueryInfo(featureStoreId=feature_store_id, queryId=query_id, request=ChatRequestBody(content=body.content, images=images_path, history=body.history, type=ChatType.ONLINE))
            ChatCache.set_query_request(query_id, feature_store_id, chat_query_info)
            ChatCache.mark_unique_inference_user(feature_store_id, ChatType.ONLINE)
            return BaseBody(data=ChatOnlineResponseBody(queryId=query_id))
        return standard_error_response(biz_constant.ERR_CHAT)

    async def fetch_response(self, body: ChatOnlineResponseBody):
        feature_store_id = self.hxd_info.featureStoreId
        info = ChatCache().get_query_info(body.queryId, feature_store_id)
        if not info:
            return standard_error_response(biz_constant.ERR_NOT_EXIST_CHAT)
        if not info.response:
            return standard_error_response(biz_constant.CHAT_STILL_IN_QUEUE)
        return BaseBody(data=info.response)

    def chat_by_agent(self, body: ChatRequestBody, t: ChatType, chat_detail: object, user_unique_id: str, query_id: str=None) -> bool:
        feature_store_id = self.hxd_info.featureStoreId
        if not query_id:
            query_id = self.generate_query_id(body.content)
        logger.info(f'[chat-request]/agent feature_store_id: {feature_store_id}, content: {body.content}, query_id: {query_id}, type:{t}')
        task = HxdTask(type=HxdTaskType.CHAT, payload=HxdTaskPayload(feature_store_id=feature_store_id, query_id=query_id, content=body.content, history=body.history, images=body.images))
        if HuixiangDouTask().updateTask(task):
            chat_query_info = ChatQueryInfo(featureStoreId=feature_store_id, queryId=query_id, request=ChatRequestBody(content=body.content, images=body.images, history=body.history), type=t, detail=chat_detail)
            ChatCache.set_query_request(query_id, feature_store_id, chat_query_info)
            ChatCache.mark_unique_inference_user(user_unique_id, t)
            return True
        return False

    def generate_query_id(self, content):
        feature_store_id = self.hxd_info.featureStoreId
        raw = feature_store_id + content[-8:] + str(time.time())
        h = hashlib.sha3_512()
        h.update(raw.encode('utf-8'))
        q = h.hexdigest()
        return q[0:8]

    def _store_images(self, images, query_id) -> List[str]:
        feature_store_id = self.hxd_info.featureStoreId
        image_store_dir = get_store_dir(feature_store_id)
        if not image_store_dir:
            logger.error(f'get store dir failed for: {feature_store_id}')
            return []
        image_store_dir += '/images/'
        os.makedirs(image_store_dir, exist_ok=True)
        ret = []
        index = 0
        for image in images:
            try:
                while len(image) % 4 != 0:
                    image += '='
                [image_format, image] = detect_base64_image_suffix(image)
                if image_format == Image.INVALID:
                    logger.error(f'invalid image format, query_id: {query_id}')
                    return []
                decoded_image = base64.b64decode(image)
            except binascii.Error:
                logger.error(f'invalid base64 encoded image, query_id: {query_id}')
                return []
            store_path = image_store_dir + query_id[-8:] + '_' + str(index) + '.' + image_format.value
            with open(store_path, 'wb') as f:
                f.write(decoded_image)
            ret.append(store_path)
        return ret

    def gen_image_store_path(self, query_id, name: str, agent: ChatType) -> Union[str, None]:
        feature_store_id = self.hxd_info.featureStoreId
        image_store_dir = get_store_dir(feature_store_id)
        if not image_store_dir:
            logger.error(f'get store dir failed for: {feature_store_id}')
            return None
        image_store_dir += '/images/'
        os.makedirs(name=image_store_dir, exist_ok=True)
        return image_store_dir + agent.name + query_id[-8:] + '_' + name

    async def case_feedback(self, body: ChatCaseFeedbackBody):
        feature_store_id = self.hxd_info.featureStoreId
        query_id = body.queryId
        query_info = ChatCache.get_query_info(query_id, feature_store_id)
        if not query_info:
            return standard_error_response(biz_constant.ERR_CHAT_CASE_FEEDBACK)
        return BaseBody() if ChatCache.update_case_feedback(feature_store_id, body.type, query_info.model_dump_json()) else standard_error_response(biz_constant.ERR_CHAT_CASE_FEEDBACK)

def detect_base64_image_suffix(base64: str) -> [Image, str]:
    if not base64 or len(base64) == 0:
        return [Image.INVALID, '']
    s = base64.split('base64,')
    if len(s) < 2:
        return [Image.INVALID, '']
    base64_prefix = s[0].lower()
    if 'data:image/jpeg;' == base64_prefix:
        return [Image.JPG, s[1]]
    if 'data:image/png;' == base64_prefix:
        return [Image.PNG, s[1]]
    if 'data:image/bmp;' == base64_prefix:
        return [Image.BMP, s[1]]
    return [Image.INVALID, '']

class LarkAgent:

    @classmethod
    async def parse_req(cls, request: Request) -> RawRequest:
        headers = dict(request.headers)
        req = RawRequest()
        req.uri = request.url.path
        req.body = await request.body()
        req.headers = {}
        for k, v in headers.items():
            if USER_AGENT.lower() == k.lower():
                req.headers[USER_AGENT] = v
            elif AUTHORIZATION.lower() == k.lower():
                req.headers[AUTHORIZATION] = v
            elif X_TT_LOGID.lower() == k.lower():
                req.headers[X_TT_LOGID] = v
            elif X_REQUEST_ID.lower() == k.lower():
                req.headers[X_REQUEST_ID] = v
            elif CONTENT_TYPE.lower() == k.lower():
                req.headers[CONTENT_TYPE] = v
            elif Content_Disposition.lower() == k.lower():
                req.headers[Content_Disposition] = v
            elif LARK_REQUEST_TIMESTAMP.lower() == k.lower():
                req.headers[LARK_REQUEST_TIMESTAMP] = v
            elif LARK_REQUEST_NONCE.lower() == k.lower():
                req.headers[LARK_REQUEST_NONCE] = v
            elif LARK_REQUEST_SIGNATURE.lower() == k.lower():
                req.headers[LARK_REQUEST_SIGNATURE] = v
        return req

    @classmethod
    def parse_rsp(cls, response: RawResponse) -> Response:
        return Response(status_code=response.status_code, content=str(response.content, UTF_8), headers=response.headers)

    @classmethod
    def get_event_handler(cls):
        return lark.EventDispatcherHandler.builder(HuixiangDouEnv.get_lark_encrypt_key(), HuixiangDouEnv.get_lark_verification_token(), lark.LogLevel.DEBUG).register_p2_im_message_receive_v1(cls._on_im_message_received).build()

    @classmethod
    def _on_im_message_received(cls, data: P2ImMessageReceiveV1):
        msg = data.event.message
        chat_id = msg.chat_id
        message_id = msg.message_id
        app_id = data.header.app_id
        app_secret = QaLibCache.get_lark_info_by_app_id(app_id)
        if not app_secret:
            logger.error(f'[lark] app_id: {app_id} not record, omit lark message callback')
            return
        client = cls._get_lark_client(app_id, app_secret)
        chat_name = cls._get_chat_name(chat_id, client)
        if not chat_name:
            logger.error(f'[lark] app_id: {app_id} get group name failed, omit lark message callback')
            return
        suffix = qalib.get_suffix_by_name(chat_name)
        if not suffix:
            logger.error(f'[lark] app_id: {app_id}, name: {chat_name} get suffix failed, omit lark message callback')
            return
        feature_store_id = QaLibCache.get_qalib_feature_store_id_by_suffix(suffix)
        if not feature_store_id:
            return
        hxd_info = QaLibCache.get_qalib_info(feature_store_id)
        if not hxd_info:
            logger.error(f'[lark] app_id: {app_id}, name: {chat_name} get feature store failed, omit lark message callback')
            return
        ChatCache.mark_agent_used(app_id, ChatType.LARK)
        if msg.root_id or msg.parent_id:
            logger.debug(f'[lark] app_id: {app_id}, name: {chat_name} got reply message, omit')
            return
        content = msg.content
        mentions = msg.mentions
        lark_content = cls._parse_lark_content(content, mentions)
        if not lark_content:
            logger.debug(f'[lark] app_id: {app_id}, name: {chat_name}, content: {content} omit')
            return
        query_id = None
        chat_svc = ChatService(None, None, hxd_info)
        if len(lark_content.images) > 0:
            query_id = chat_svc.generate_query_id(lark_content.content)
            for index in range(len(lark_content.images)):
                image_store_path = chat_svc.gen_image_store_path(query_id, str(index), ChatType.LARK)
                if cls._store_image(client, message_id, lark_content.images[index], image_store_path):
                    lark_content.images[index] = image_store_path
        chat_detail = LarkChatDetail(appId=app_id, appSecret=app_secret, messageId=msg.message_id)
        unique_id = data.event.sender.sender_id.open_id + '@' + chat_id
        chat_svc.chat_by_agent(lark_content, ChatType.LARK, chat_detail, unique_id, query_id)

    @classmethod
    def _get_chat_name(cls, chat_id: str, client: lark.client) -> Union[str, None]:
        request = GetChatRequest.builder().chat_id(chat_id).build()
        response = client.im.v1.chat.get(request)
        if not response.success():
            logger.error(f'[lark] get chat: {chat_id} info failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}')
            return None
        try:
            return response.data.name
        except:
            logger.error(f'[lark] get chat: {chat_id} name failed, data: {response.data}')
            return None

    @classmethod
    def _get_lark_client(cls, app_id: str, app_secret: str) -> lark.client:
        return lark.Client.builder().app_id(app_id).app_secret(app_secret).log_level(HuixiangDouEnv.get_lark_log_level()).build()

    @classmethod
    def _parse_lark_content(cls, content: str, mentions: List[MentionEvent]) -> Union[ChatRequestBody, None]:
        if not content or len(content) == 0:
            return None
        lark_json = json.loads(content)
        content_type = LarkContentType.NORMAL_TEXT
        if 'text' in lark_json:
            text = lark_json.get('text')
            if '@_user' in text:
                content_type = cls._get_content_type_when_at_user_exists(mentions)
            elif '@_all' in text:
                content_type = LarkContentType.AT_ALL_TEXT
        elif len(lark_json) == 1 and 'image_key' in lark_json:
            content_type = LarkContentType.IMAGE
        else:
            content_type = LarkContentType.OTHER
        process_flag = cls._check_should_process(content_type)
        if not process_flag:
            logger.debug(f'[lark] content: {content} has content_type: {content_type}, omit')
            return None
        if content_type == LarkContentType.IMAGE:
            image_key = lark_json.get('image_key')
            return ChatRequestBody(images=[image_key])
        else:
            text = lark_json.get('text')
            if content_type != LarkContentType.NORMAL_TEXT:
                text = re.sub('@_user_\\d+', '', text)
                text = re.sub('@_all\\d', '', text)
            return ChatRequestBody(content=text)

    @classmethod
    def _check_should_process(cls, t: LarkContentType) -> bool:
        return t == LarkContentType.AT_BOT_TEXT or t == LarkContentType.NORMAL_TEXT or t == LarkContentType.AT_ALL_TEXT or (t == LarkContentType.IMAGE)

    @classmethod
    def _get_content_type_when_at_user_exists(cls, mentions: List[MentionEvent]) -> LarkContentType:
        if not mentions or len(mentions) == 0:
            return LarkContentType.AT_OTHER_PERSON_TEXT
        for item in mentions:
            if not item.id.user_id or len(item.id.user_id) == 0:
                return LarkContentType.AT_BOT_TEXT
        return LarkContentType.AT_OTHER_PERSON_TEXT

    @classmethod
    def _store_image(cls, client: lark.client, message_id: str, image_key: str, path: str) -> bool:
        body = GetMessageResourceRequest.builder().message_id(message_id).file_key(image_key).build()
        response = client.im.v1.message_resource.get(body)
        if not response.success():
            logger.error(f'[lark] get image: {image_key} info failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}')
            return False
        if response.file:
            with open(path, mode='wb') as fout:
                fout.write(response.file.read())
            return True
        logger.error(f'[lark] get image: {image_key} stream empty, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}')
        return False

    @classmethod
    async def response_callback(cls, chat_info: ChatQueryInfo) -> bool:
        if not chat_info.detail:
            logger.error(f'[lark] invalid lark detail to send response, chat_info: {chat_info.model_dump()}')
            return False
        if chat_info.response.code != 0:
            logger.info(f'[lark] HuixiangDou inference error, detail: {chat_info.response.model_dump()}')
            return True
        lark_detail = json.dumps(chat_info.detail)
        lark_detail = LarkChatDetail(**json.loads(lark_detail))
        client = cls._get_lark_client(lark_detail.appId, lark_detail.appSecret)
        content_body = ReplyMessageRequestBody.builder().content(json.dumps({'text': chat_info.response.text})).msg_type('text').reply_in_thread(False).build()
        reply_body = ReplyMessageRequest.builder().message_id(lark_detail.messageId).request_body(content_body).build()
        response = await client.im.v1.message.areply(reply_body)
        if not response.success():
            logger.error(f'[lark] response: {chat_info.model_dump()} failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}')
            return False
        return True

def check_endpoint_update(info: QalibInfo):
    update = False
    if not info.wechat or info.wechat.onMessageUrl.endswith('wechat'):
        info.wechat = Wechat(onMessageUrl=get_wechat_on_message_url(info.suffix))
        update = True
    else:
        wechat_message_url = get_wechat_on_message_url(info.suffix)
        if info.wechat.onMessageUrl != wechat_message_url:
            info.wechat.onMessageUrl = wechat_message_url
            update = True
    if info.lark:
        lark_event_url = get_lark_on_message_url()
        if info.lark.eventUrl != lark_event_url:
            info.lark.eventUrl = lark_event_url
            update = True
    if update:
        QaLibCache.set_qalib_info(info.featureStoreId, info)


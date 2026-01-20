# Cluster 46

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


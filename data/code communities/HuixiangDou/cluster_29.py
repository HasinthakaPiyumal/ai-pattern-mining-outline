# Cluster 29

def handle_task_add_doc_response(response: HxdTaskResponse):
    """update qalib's status from huixiangdou response's code.

    :param response:
    :return:
    """
    logger.info('do task: add doc')
    fid = response.feature_store_id
    name = biz_const.RDS_KEY_QALIB_INFO
    files_state = response.files_state
    o = r.hget(name=name, key=fid)
    if not o:
        logger.error(f"can't find {name}:{fid} in redis.")
        return
    qalib_info = QalibInfo(**json.loads(o))
    qalib_info.status = biz_const.HXD_PIPELINE_QALIB_CREATE_SUCCESS if response.code == 0 else response.code
    qalib_info.status_desc = response.status
    qalib_info.filesState = files_state
    r.hset(name=name, key=fid, value=qalib_info.model_dump_json())
    logger.info(f"do task={response.type} with fid={response.feature_store_id}'s result: {response.code}-{response.status}")

def handle_task_update_sample_response(response: HxdTaskResponse):
    """update sample's confirm status from response's code.

    :param response:
    :return:
    """
    logger.info('do task: update sample')
    name = biz_const.RDS_KEY_SAMPLE_INFO
    fid = response.feature_store_id
    o = r.hget(name=name, key=fid)
    if not o:
        logger.error(f"can't find {name}:{fid} in redis")
        return
    sample = QalibSample(**json.loads(o))
    sample.confirmed = True if response.code == 0 else False
    r.hset(name=name, key=fid, value=sample.model_dump_json())

def handle_task_update_pipeline_response(response: HxdTaskResponse):
    logger.info('do task: update pipeline')
    name = biz_const.RDS_KEY_PIPELINE
    o = r.hget(name=name, key=response.feature_store_id)
    if not o:
        logger.error(f"can't find {name}:{response.feature_store_id} in redis")
        return
    pipeline = Pipeline(**json.loads(o))
    pipeline.status = response.status
    pipeline.code = response.code
    pipeline.confirmed = True
    pipeline.success = True if response.code == 0 else False
    r.hset(name=name, key=response.feature_store_id, value=pipeline.model_dump_json())

class QaLibService:

    def __init__(self, request: Request, response: Response, hxd_info: QalibInfo):
        self.request = request
        self.response = response
        self.hxd_info = hxd_info

    @classmethod
    def get_existed_docs(cls, feature_store_id) -> List:
        o = r.hget(name=biz_const.RDS_KEY_QALIB_INFO, key=feature_store_id)
        if not o:
            return []
        qalib_info = QalibInfo(**json.loads(o))
        return qalib_info.docs

    async def info(self) -> BaseBody:
        return BaseBody(data=self.hxd_info)

    async def add_docs(self, files: List[UploadFile]=File(...)):
        feature_store_id = self.hxd_info.featureStoreId
        name = self.hxd_info.name
        logger.info(f'start to add docs for qalib: {name}')
        store_dir = get_store_dir(feature_store_id)
        if not files or not store_dir:
            return BaseBody()
        ret = AddDocsRes(errors=[])
        docs = self.get_existed_docs(feature_store_id)
        total_bytes = int(self.request.headers.get('content-length'))
        if total_bytes > biz_const.HXD_ADD_DOCS_ONCE_MAX:
            return standard_error_response(biz_const.ERR_QALIB_ADD_DOCS_ONCE_MAX)
        write_size = 0
        for file in files:
            if file.filename and len(file.filename.encode('utf-8')) > 255:
                logger.error(f'filename: {file.filename} too long, maximum 255 bytes, omit current filename')
                ret.errors.append(AddDocError(fileName=file.filename, reason='filename is too long'))
                continue
            with open(os.path.join(store_dir, file.filename), 'wb') as f:
                while True:
                    chunk = await file.read(32768)
                    if not chunk:
                        break
                    f.write(chunk)
                    write_size += len(chunk)
                    progress = write_size / total_bytes * 100
                    print(f'\rQalib({name}) total process: {progress:.2f}%', end='')
            docs.append(file.filename)
            await file.close()
        if not QaLibCache().update_qalib_docs(feature_store_id, docs, store_dir):
            return BaseBody()
        if not HuixiangDouTask().updateTask(HxdTask(type=HxdTaskType.ADD_DOC, payload=HxdTaskPayload(name=name, feature_store_id=feature_store_id, file_list=docs, file_abs_base=store_dir))):
            return BaseBody()
        ret.docBase = store_dir
        ret.docs = docs
        return BaseBody(data=ret)

    async def delete_docs(self, body: QalibDeleteDoc):
        feature_store_id = self.hxd_info.featureStoreId
        name = self.hxd_info.name
        logger.info(f'start to delete docs for qalib: {name}')
        store_dir = get_store_dir(feature_store_id)
        for filename in body.filenames:
            path = os.path.join(store_dir, filename)
            if not os.path.exists(path):
                logger.warn(f'qalib: {name} has no file named {filename} to delete.')
                continue
            if path.startswith('.'):
                continue
            try:
                os.remove(path)
            except OSError as e:
                logger.error(f'qalib: error: {e} when removing {path}')
        filenames = set(os.listdir(store_dir))
        left_filenames = list(filenames - set(body.filenames))
        if not QaLibCache().rewrite_qalib_docs(feature_store_id, left_filenames, store_dir):
            return BaseBody()
        if not HuixiangDouTask().updateTask(HxdTask(type=HxdTaskType.ADD_DOC, payload=HxdTaskPayload(name=name, feature_store_id=feature_store_id, file_list=left_filenames, file_abs_base=store_dir))):
            return BaseBody()
        ret = AddDocsRes(errors=[])
        ret.docBase = store_dir
        ret.docs = left_filenames
        return BaseBody(data=ret)

    async def get_sample_info(self):
        sample_info = QaLibCache.get_sample_info(self.hxd_info.featureStoreId)
        return BaseBody(data=sample_info)

    async def update_sample_info(self, body: QalibPositiveNegative):
        name = self.hxd_info.name
        feature_store_id = self.hxd_info.featureStoreId
        positives = body.positives
        negatives = body.negatives
        qalib_sample = QalibSample(name=name, featureStoreId=feature_store_id, positives=positives, negatives=negatives, confirmed=False)
        QaLibCache.set_sample_info(feature_store_id, qalib_sample)
        if not HuixiangDouTask().updateTask(HxdTask(type=HxdTaskType.UPDATE_SAMPLE, payload=HxdTaskPayload(name=name, feature_store_id=feature_store_id, positive=positives, negative=negatives))):
            return BaseBody()
        return await self.get_sample_info()

    async def integrate_lark(self, body: IntegrateLarkBody):
        feature_store_id = self.hxd_info.featureStoreId
        info = QaLibCache.get_qalib_info(feature_store_id)
        if not info:
            return standard_error_response(biz_const.ERR_QALIB_INFO_NOT_FOUND)
        if info.lark:
            info.lark.appId = body.appId
            info.lark.appSecret = body.appSecret
        else:
            info.lark = Lark(appId=body.appId, appSecret=body.appSecret, encryptKey=HuixiangDouEnv.get_lark_encrypt_key(), verificationToken=HuixiangDouEnv.get_lark_verification_token(), eventUrl=get_lark_on_message_url())
        QaLibCache.set_qalib_info(feature_store_id, info)
        QaLibCache.set_lark_info(body.appId, body.appSecret)
        return BaseBody(data=info.lark)

    async def integrate_web_search(self, body: IntegrateWebSearchBody):
        feature_store_id = self.hxd_info.featureStoreId
        info = QaLibCache.get_qalib_info(feature_store_id)
        if not info:
            return standard_error_response(biz_const.ERR_QALIB_INFO_NOT_FOUND)
        info.webSearch = WebSearch(token=body.webSearchToken)
        task = HxdTask(type=HxdTaskType.UPDATE_PIPELINE, payload=HxdTaskPayload(feature_store_id=feature_store_id, name=self.hxd_info.name, web_search_token=body.webSearchToken))
        if HuixiangDouTask().updateTask(task):
            QaLibCache.set_qalib_info(feature_store_id, info)
            return BaseBody()
        return standard_error_response(biz_const.ERR_INFO_UPDATE_FAILED)

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

class ChatCache:

    def __init__(self):
        pass

    @classmethod
    def set_query_request(cls, query_id: str, feature_store_id: str, info: ChatQueryInfo):
        cls._set_query_info(query_id, feature_store_id, info)

    @classmethod
    def set_query_response(cls, query_id: str, feature_store_id: str, response: ChatResponse) -> Union[ChatQueryInfo, None]:
        q = cls.get_query_info(query_id, feature_store_id)
        if not q:
            return None
        q.response = response
        cls._set_query_info(query_id, feature_store_id, q)
        return q

    @classmethod
    def get_query_info(cls, query_id: str, feature_store_id: str) -> Union[ChatQueryInfo, None]:
        key = biz_constant.RDS_KEY_QUERY_INFO + ':' + feature_store_id
        field = query_id
        o = r.hget(key, field)
        if not o or len(o) == 0:
            logger.error(f'feature_store_id: {feature_store_id} get query: {query_id} empty, omit')
            return None
        return ChatQueryInfo(**json.loads(o))

    @classmethod
    def mget_query_info(cls, query_id_list: List[str], feature_store_id: str) -> Union[List[ChatQueryInfo], None]:
        key = biz_constant.RDS_KEY_QUERY_INFO + ':' + feature_store_id
        o = r.hmget(key, query_id_list)
        if not o or len(o) == 0:
            logger.error(f'feature_store_id: {feature_store_id} mget: {query_id_list} empty, omit')
            return None
        ret = []
        for item in o:
            ret.append(ChatQueryInfo(**json.loads(item)))
        return ret

    @classmethod
    def _set_query_info(cls, query_id: str, feature_store_id: str, info: ChatQueryInfo):
        key = biz_constant.RDS_KEY_QUERY_INFO + ':' + feature_store_id
        field = query_id
        r.hset(key, field, info.model_dump_json())

    @classmethod
    def update_case_feedback(cls, feature_store_id: str, case_type: ChatCaseType, feedback: str) -> bool:
        try:
            name = f'{biz_constant.RDS_KEY_FEEDBACK_CASE}:{case_type}:{feature_store_id}'
            r.rpush(name, feedback)
            return True
        except Exception as e:
            logger.error(f'{e}')
            return False

    @classmethod
    def record_query_id_to_fetch(cls, feature_store_id: str, query_id: str):
        key = biz_constant.RDS_KEY_QUERY_ID_TO_FETCH + ':' + feature_store_id
        r.hset(key, query_id, 1)
        r.expire(key, biz_constant.HXD_CHAT_TTL)

    @classmethod
    def mget_query_id_to_fetch(cls, feature_store_id: str) -> List[str]:
        key = biz_constant.RDS_KEY_QUERY_ID_TO_FETCH + ':' + feature_store_id
        o = r.hgetall(key)
        if not o or len(o) == 0:
            return []
        ret = []
        for i in o.keys():
            ret.append(i)
        return ret

    @classmethod
    def mark_query_id_complete(cls, feature_store_id: str, query_id_list: List[str]):
        if len(query_id_list) == 0:
            return
        key = biz_constant.RDS_KEY_QUERY_ID_TO_FETCH + ':' + feature_store_id
        for item in query_id_list:
            r.hdel(key, item)

    @classmethod
    def mark_agent_used(cls, agent_identifier: str, agent: ChatType):
        field = agent_identifier
        if agent == ChatType.LARK:
            key = biz_constant.RDS_KEY_AGENT_LARK_USED
        else:
            key = biz_constant.RDS_KEY_AGENT_WECHAT_USED
        r.hset(key, field, 1)

    @classmethod
    def hlen_agent_used(cls, agent: ChatType) -> int:
        if agent == ChatType.LARK:
            key = biz_constant.RDS_KEY_AGENT_LARK_USED
        else:
            key = biz_constant.RDS_KEY_AGENT_WECHAT_USED
        o = r.hlen(key)
        return o

    @classmethod
    def mark_monthly_active(cls, feature_store_id: str):
        today_month = time_util.get_month_time_str(datetime.now())
        key = biz_constant.RDS_KEY_QALIB_ACTIVE + ':' + today_month
        r.hset(key, feature_store_id, 1)

    @classmethod
    def get_monthly_active(cls) -> int:
        today_month = time_util.get_month_time_str(datetime.now())
        key = biz_constant.RDS_KEY_QALIB_ACTIVE + ':' + today_month
        o = r.hlen(key)
        return o

    @classmethod
    def add_inference_number(cls):
        key = biz_constant.RDS_KEY_TOTAL_INFERENCE_NUMBER
        r.incr(key)

    @classmethod
    def get_inference_number(cls) -> int:
        key = biz_constant.RDS_KEY_TOTAL_INFERENCE_NUMBER
        o = r.get(key)
        return o

    @classmethod
    def mark_unique_inference_user(cls, user_identifier: str, agent: ChatType):
        key = biz_constant.RDS_KEY_USER_INFERENCE
        field = user_identifier + '@' + agent.name
        r.sadd(key, field)

    @classmethod
    def get_unique_inference_user_number(cls) -> int:
        key = biz_constant.RDS_KEY_USER_INFERENCE
        o = r.scard(key)
        return o


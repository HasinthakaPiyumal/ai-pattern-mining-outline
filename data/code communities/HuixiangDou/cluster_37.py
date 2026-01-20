# Cluster 37

def add_access_info(name, value) -> bool:
    """add new access info to access:info db.

    :param name:
    :param value:
    :return:
    """
    return False if 1 != r.hset(name=biz_const.RDS_KEY_LOGIN, key=name, value=value) else True

def del_access_info(name) -> bool:
    """del new access info from access:info db.

    :param name:
    :return:
    """
    return True if 1 == r.hdel(biz_const.RDS_KEY_LOGIN, name) else False

def _create_qa_lib(name, hashed_pass, feature_store_id) -> bool:
    """
    1. init access info
    2. init qalib info
    :param name:
    :param hashed_pass:
    :param feature_store_id:
    :return:
    """
    try:
        if not add_access_info(name, AccessInfo(hashpass=hashed_pass, featureStoreId=feature_store_id).model_dump_json()):
            return False
        suffix = gen_suffix(feature_store_id)
        if not QaLibCache().init_qalib_info(feature_store_id, biz_const.HXD_QALIB_STATUS_INIT, name, suffix):
            if not del_access_info(name):
                logger.error(f'del access info by {name} failed')
            return False
        QaLibCache().set_suffix_to_qalib(suffix, feature_store_id)
        return True
    except Exception as e:
        logger.error(f'[create] create name: {name} failed: {e}')
        if not del_access_info(name):
            logger.error(f'del access info by {name} failed')
        if not QaLibCache().del_qalib_info(feature_store_id):
            logger.error(f'del qalib info by {name} failed')
        return False

def gen_suffix(feature_store_id: str) -> str:
    length = biz_const.HXD_FEATURE_STORE_SUFFIX_LENGTH
    if len(feature_store_id) <= length:
        return feature_store_id
    return feature_store_id[-length:]

def gen_random_string(length=4) -> str:
    """
    :param length: random string's length
    :return: a string with the given length, includes only A-Za-z0-9
    """
    chars = string.ascii_letters + string.digits
    return ''.join((random.choice(chars) for _ in range(length)))

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


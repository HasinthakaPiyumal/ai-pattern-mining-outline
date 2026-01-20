# Cluster 38

class LoginService:

    def __init__(self, login: LoginBody, request: Request, response: Response):
        self.name = login.name
        self.password = login.password
        self.response = response
        self.request = request

    def _set_cookie(self, cookie_key, *jwt_payloads):
        self.response.set_cookie(key=cookie_key, value=str.gen_jwt(jwt_payloads[0][0], jwt_payloads[0][1], int(round(time.time() * 1000) + 604800000)), max_age=604800, expires=604800, secure=HuixiangDouEnv.get_cookie_secure(), samesite=HuixiangDouEnv.get_cookie_samesite())

    async def login(self):
        if not self.name or len(self.name) < 8:
            logger.error(f'login name={self.name} not valid.')
            return BaseBody(msg=biz_const.ERR_ACCESS_LOGIN.get('msg'), msgCode=biz_const.ERR_ACCESS_LOGIN.get('code'))
        o = r.hget(name=biz_const.RDS_KEY_LOGIN, key=self.name)
        gen_hashed_pass = bcrypt.hash(self.password)
        if not o:
            feature_store_id = str.gen_random_string()
            if not _create_qa_lib(self.name, gen_hashed_pass, feature_store_id):
                self.response.delete_cookie(key=biz_const.HXD_COOKIE_KEY)
                return BaseBody(msg=biz_const.ERR_ACCESS_CREATE.get('msg'), msgCode=biz_const.ERR_ACCESS_CREATE.get('code'))
            self._set_cookie(biz_const.HXD_COOKIE_KEY, [feature_store_id, self.name])
            return BaseBody(data={'exist': False, 'featureStoreId': feature_store_id})
        else:
            access_info = AccessInfo(**json.loads(bytes.decode(o)))
            if bcrypt.verify(self.password, access_info.hashpass):
                feature_store_id = access_info.featureStoreId
            else:
                return BaseBody(msg=biz_const.ERR_ACCESS_LOGIN.get('msg'), msgCode=biz_const.ERR_ACCESS_LOGIN.get('code'))
        self._set_cookie(biz_const.HXD_COOKIE_KEY, [feature_store_id, self.name])
        return BaseBody(data={'exist': True, 'featureStoreId': feature_store_id})

def gen_jwt(feature_store_id: str, qa_name: str, expire: int) -> str:
    """
    :param feature_store_id:
    :param qa_name: 知识库名称
    :param expire: 过期时间 unix 时间戳
    :return: jwt
    """
    payload = {'iat': time.time(), 'jti': feature_store_id, 'qa_name': qa_name, 'exp': expire}
    token = jwt.encode(payload, HuixiangDouEnv.get_jwt_secret(), algorithm='HS256')
    return token

def get_hxd_token_by_cookie(cookies) -> Union[HxdToken, None]:
    return HxdToken(**str_util.parse_jwt(cookies.get(biz_const.HXD_COOKIE_KEY))) if cookies and cookies.get(biz_const.HXD_COOKIE_KEY) else None

def parse_jwt(token: str) -> dict:
    hxd_token = jwt.decode(token, HuixiangDouEnv.get_jwt_secret(), algorithms='HS256')
    return hxd_token

def _get_hxd_token_by_cookie(cookies) -> Union[HxdToken, None]:
    return HxdToken(**str_util.parse_jwt(cookies.get(biz_const.HXD_COOKIE_KEY))) if cookies and cookies.get(biz_const.HXD_COOKIE_KEY) else None


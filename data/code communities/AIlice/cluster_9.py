# Cluster 9

@app.route('/proxy', methods=['GET', 'HEAD'])
def proxy():
    href = unquote(request.args.get('href'))
    logger.debug(f'Proxy request for {href}')
    res = context.CurrentSession().Proxy(href, request.method)
    meta = next(res)
    if 'variable' == meta['type']:
        with tempfile.NamedTemporaryFile(mode='bw', delete=True) as temp:
            temp.write(meta['data'].data)
            temp.flush()
            if request.method == 'HEAD':
                response = make_response('')
                response.headers['Content-Type'] = {'AImage': 'image/jpeg', 'AVideo': 'video/mp4'}[type(meta['data']).__name__]
            else:
                response = send_file(os.path.abspath(temp.name))
    else:
        try:
            responseInfo = meta['responseInfo']

            def gen():
                yield from res
            contentType = responseInfo['headers'].get('Content-Type', '')
            response = Response(gen(), status=responseInfo['status_code'], content_type=contentType)
            for key, value in responseInfo['headers'].items():
                if key.lower() not in ('content-encoding', 'content-length', 'transfer-encoding', 'connection'):
                    response.headers[key] = value
            if contentType.lower() in ('image/svg+xml', 'application/svg+xml'):
                response.headers['Content-Type'] = 'image/svg+xml'
                response.headers['Content-Disposition'] = 'inline'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = '*'
        except requests.exceptions.RequestException as e:
            error_msg = f'Error fetching the URL: {e}'
            logger.error(error_msg)
            return (error_msg, 500)
    isLocalFile = os.path.exists(href) if href.startswith('/') or href.startswith('file://') or ':/' in href else False
    if isLocalFile:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    else:
        response.headers['Cache-Control'] = 'public, max-age=60'
    return response

def gen():
    yield from res

class AImageLocation(BaseModel):
    urlOrPath: str

    def IsURL(self, ident: str) -> bool:
        return urlparse(ident).scheme != ''

    def GetImage(self, ident: str, proxy=None) -> Image:
        if proxy is None:
            if self.IsURL(ident):
                response = requests.get(ident)
                imageBytes = io.BytesIO(response.content)
                return Image.open(imageBytes)
            else:
                return Image.open(ident)
        else:
            response = proxy(ident, 'GET')
            _ = next(response)
            imageBytes = io.BytesIO()
            for chunk in response:
                imageBytes.write(chunk)
            return Image.open(imageBytes)

    @classmethod
    def FromJson(cls, data):
        return cls(urlOrPath=data['urlOrPath'])

    def ToJson(self):
        return {'type': 'AImageLocation', 'urlOrPath': self.urlOrPath}

    def Standardize(self, proxy=None):
        image = self.GetImage(self.urlOrPath, proxy)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        imageByte = io.BytesIO()
        image.save(imageByte, format='JPEG')
        return AImage(data=imageByte.getvalue())

class AVideoLocation(BaseModel):
    urlOrPath: str

    def IsURL(self, ident: str) -> bool:
        return urlparse(ident).scheme != ''

    def GetVideo(self, ident: str, proxy=None):
        if proxy is None:
            if self.IsURL(ident):
                response = requests.get(ident)
                videoBytes = io.BytesIO(response.content)
                return videoBytes.getvalue()
            else:
                with open(ident, 'rb') as f:
                    videoBytes = io.BytesIO(f.read())
                    return videoBytes.getvalue()
        else:
            response = proxy(ident, 'GET')
            _ = next(response)
            videoBytes = io.BytesIO()
            for chunk in response:
                videoBytes.write(chunk)
            return videoBytes.getvalue()

    @classmethod
    def FromJson(cls, data):
        return cls(urlOrPath=data['urlOrPath'])

    def ToJson(self):
        return {'type': 'AVideoLocation', 'urlOrPath': self.urlOrPath}

    def Standardize(self, proxy=None):
        return AVideo(data=ConvertVideoFormat(self.GetVideo(self.urlOrPath, proxy), 'mp4'))

@app.route('/proxy', methods=['GET', 'HEAD'])
def proxy():
    href = unquote(request.args.get('href'))
    var = context[currentSession]['processor'].interpreter.env.get(href, None)
    if var and type(var).__name__ in ['AImage', 'AVideo']:
        with tempfile.NamedTemporaryFile(mode='bw', delete=True) as temp:
            temp.write(var.data)
            temp.flush()
            if request.method == 'HEAD':
                response = make_response('')
                response.headers['Content-Type'] = {'AImage': 'image/jpeg', 'AVideo': 'video/mp4'}[type(var).__name__]
            else:
                response = send_file(os.path.abspath(temp.name))
    else:
        computer = context[currentSession]['processor'].services.GetClient(config.services['computer']['addr'])
        try:
            r = computer.Proxy(href, request.method)
            responseInfo = next(r)

            def gen():
                yield from r
            contentType = responseInfo['headers'].get('Content-Type', '')
            response = Response(gen(), status=responseInfo['status_code'], content_type=contentType)
            for key, value in responseInfo['headers'].items():
                if key.lower() not in ('content-encoding', 'content-length', 'transfer-encoding', 'connection'):
                    response.headers[key] = value
            if contentType.lower() in ('image/svg+xml', 'application/svg+xml'):
                response.headers['Content-Type'] = 'image/svg+xml'
                response.headers['Content-Disposition'] = 'inline'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = '*'
        except requests.exceptions.RequestException as e:
            return (f'Error fetching the URL: {e}', 500)
    isLocalFile = os.path.exists(href) if href.startswith('/') or href.startswith('file://') or ':/' in href else False
    if isLocalFile:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    else:
        response.headers['Cache-Control'] = 'public, max-age=60'
    return response


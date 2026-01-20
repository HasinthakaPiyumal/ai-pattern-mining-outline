# Cluster 26

class AComputer:

    def __init__(self):
        self.lock = threading.Lock()
        if 0 == len(requirements):
            self.clicks = {'click': pyautogui.click, 'double-click': pyautogui.doubleClick, 'right-click': pyautogui.rightClick, 'middle': pyautogui.middleClick}
            self.reader = easyocr.Reader(['en'])
        return

    def ModuleInfo(self):
        return {'NAME': 'computer', 'ACTIONS': {'SCREENSHOT': {'func': 'ScreenShot', 'prompt': 'Take a screenshot of the current screen.', 'type': 'primary'}, 'LOCATEANDCLICK': {'func': 'LocateAndClick', 'prompt': "Locate the control containing a piece of text on the screenshot and click on it. clickType is a string, and its value can only be one of 'click', 'double-click', 'right-click' or 'middle'.", 'type': 'primary'}, 'LOCATEANDSCROLL': {'func': 'LocateAndScroll', 'prompt': 'Move to the position marked by the text and scroll the mouse wheel.', 'type': 'primary'}, 'TYPEWRITE': {'func': 'TypeWrite', 'prompt': 'Simulate keyboard input for the string. Please ensure that the focus has been moved to the location where input is expected.', 'type': 'primary'}, 'READ-IMAGE': {'func': 'ReadImage', 'prompt': 'Read the content of an image file into a variable.', 'type': 'primary'}, 'WRITE-IMAGE': {'func': 'WriteImage', 'prompt': 'Write a variable of image type into a file.', 'type': 'primary'}}}

    def Locate(self, txt: str):
        image = ImageGrab.grab()
        results = self.reader.readtext(numpy.array(image.convert('L')), slope_ths=0.0, ycenter_ths=0.0, width_ths=0.0)
        for detection in results:
            bbox = detection[0]
            text = detection[1]
            if txt in text:
                x, y = (int((bbox[0][0] + bbox[2][0]) * 0.5), int((bbox[0][1] + bbox[2][1]) * 0.5))
                return (x, y, text)
        return None

    def ScreenShot(self) -> AImage:
        with self.lock:
            imageByte = io.BytesIO()
            ImageGrab.grab().save(imageByte, format='JPEG')
            return AImage(data=imageByte.getvalue())

    def LocateAndClick(self, txt: str, clickType: str) -> str:
        with self.lock:
            if 0 != len(requirements):
                return f'python package(s) {[x for x in requirements]} not found. Please install it before using this feature.'
            if clickType not in self.clicks:
                return f"LOCATEANDCLICK ERROR. clickType: {clickType} can only be one of 'click', 'double-click', 'right-click' or 'middle'."
            ret = self.Locate(txt)
            if None != ret:
                x, y, text = ret
                pyautogui.moveTo(x, y, duration=0.5)
                self.clicks[clickType]()
                return f"'''{text}''' at {x},{y} is clicked."
            else:
                return f"'''{txt}''' not found. It may be because the text has been segmented into different boxes by the OCR software. Please try a shorter and distinctive substring."

    def LocateAndScroll(self, txt: str, clicks: float) -> str:
        with self.lock:
            if 0 != len(requirements):
                return f'python package(s) {[x for x in requirements]} not found. Please install it before using this feature.'
            ret = self.Locate(txt)
            if None != ret:
                x, y, text = ret
                pyautogui.moveTo(x, y, duration=0.5)
                pyautogui.scroll(clicks)
                return f'The mouse wheel has scrolled {clicks} times.'
            else:
                return f"'''{txt}''' not found. It may be because the text has been segmented into different boxes by the OCR software. Please try a shorter and distinctive substring."

    def TypeWrite(self, txt: str) -> str:
        with self.lock:
            if 0 != len(requirements):
                return f'python package(s) {[x for x in requirements]} not found. Please install it before using this feature.'
            pyautogui.typewrite(txt)
            return f"'''{txt}''' the string has already been typed."

    def ReadImage(self, path: str) -> AImage:
        try:
            return AImageLocation(urlOrPath=path).Standardize()
        except Exception as e:
            print('ReadImage() excetption: ', e)
        return AImage(data=None)

    def WriteImage(self, image: AImage, path: str) -> str:
        try:
            Image.open(io.BytesIO(image.data)).save(path)
            return f'The image has been written to {path}.'
        except Exception as e:
            print('WriteImage() excetption: ', e)
            return f'WriteImage() excetption: {str(e)}'
        return

    def WriteFile(self, data: bytes, path: str) -> str:
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, 'wb') as file:
                file.write(data)
            return f'The file has been written to {path}.'
        except Exception as e:
            print('WriteFile() excetption: ', e)
            return f'WriteFile() excetption: {str(e)}'
        return

    def Proxy(self, href: str, method: str, headers: dict={}, body: dict={}, params: dict={}) -> typing.Generator:
        if os.path.exists(href):
            filePath = os.path.abspath(href)
            fileSize = os.path.getsize(filePath)
            fileName = os.path.basename(filePath)
            contentType, encoding = mimetypes.guess_type(filePath)
            if contentType is None:
                contentType = 'application/octet-stream'
            startByte = 0
            endByte = fileSize - 1
            statusCode = 200
            if headers and 'Range' in headers:
                rangeHeader = headers['Range']
                rangeMatch = re.match('bytes=(\\d+)-(\\d*)', rangeHeader)
                if rangeMatch:
                    startByte = int(rangeMatch.group(1))
                    if rangeMatch.group(2):
                        endByte = min(int(rangeMatch.group(2)), fileSize - 1)
                    statusCode = 206
            responseHeaders = {'Content-Type': contentType, 'Content-Length': str(endByte - startByte + 1), 'Accept-Ranges': 'bytes', 'Content-Disposition': f"""inline; filename="{urllib.parse.quote(fileName)}"; filename*=UTF-8''{urllib.parse.quote(fileName)}""", 'Last-Modified': datetime.datetime.fromtimestamp(os.stat(filePath).st_mtime, tz=datetime.timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')}
            if statusCode == 206:
                responseHeaders['Content-Range'] = f'bytes {startByte}-{endByte}/{fileSize}'
            responseInfo = {'status_code': statusCode, 'headers': responseHeaders}
            yield responseInfo
            if method.upper() != 'HEAD':

                def content_generator():
                    with open(filePath, 'rb') as file:
                        if startByte > 0:
                            file.seek(startByte)
                        bytesToRead = endByte - startByte + 1
                        bytesRead = 0
                        while bytesRead < bytesToRead:
                            chunkSize = min(262144, bytesToRead - bytesRead)
                            chunk = file.read(chunkSize)
                            if not chunk:
                                break
                            bytesRead += len(chunk)
                            yield chunk
                yield from content_generator()
        else:
            req = requests.request(method=method, url=href, headers=headers, data=body, params=params, stream=True)
            responseInfo = {'status_code': req.status_code, 'headers': dict(req.headers)}
            yield responseInfo
            if method.upper() != 'HEAD':

                def content_generator():
                    try:
                        for chunk in req.iter_content(chunk_size=262144):
                            if chunk:
                                yield chunk
                    finally:
                        req.close()
                yield from content_generator()

class AConversations:

    def __init__(self, proxy):
        self.proxy = proxy
        self.conversations: list[dict] = []
        return

    def Add(self, role: str, msg: str, env: dict[str, Any], entry: bool=False):
        msg = '<EMPTY MSG>' if '' == msg else msg
        record = {'role': role, 'time': time.time(), 'entry': entry, 'msg': msg, 'attachments': []}
        if role in ['USER', 'SYSTEM']:
            matches = re.findall('```(\\w*)\\n([\\s\\S]*?)```', msg)
            vars = []
            for language, code in matches:
                varName = f'code_{language}_{str(random.randint(0, 10000))}'
                env[varName] = code
                vars.append(varName)
            if 0 < len(vars):
                record['msg'] += f'\nSystem notification: The code snippets within the triple backticks in this message have been saved as variables, in accordance with their order in the text, the variable names are as follows: {vars}\n'
            matches = [m for m in re.findall('(!\\[([^\\]]*?)\\]\\((.*?)\\)(?:<([a-zA-Z0-9_\\-&]+)>)?)', msg)]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.ProcessMultimodalTags, m, param, label, env) for m, txt, param, label in matches]
                for future, match in zip(concurrent.futures.as_completed(futures), matches):
                    try:
                        m, txt, param, label = match
                        result = future.result()
                        if isinstance(result, Exception):
                            msgNew = msg.replace(m, f'{m}\n(System notification: Unable to get multimodal content: {e})')
                            record['msg'] = msgNew
                        elif None != result:
                            record['attachments'].append(result)
                    except Exception as e:
                        record['msg'] += f'\nSystem notification: Exception encountered while processing multimodal tags: {str(e)}'
        self.conversations.append(record)
        return

    def ProcessMultimodalTags(self, m, param, label, env):
        if '&' == label:
            if '' == param or param not in env:
                raise ValueError(f'variable name ({param}) not defined.')
            return {'type': typeInfo[type(env[param])]['modal'], 'tag': m, 'content': env[param].Standardize()}
        elif '' != label:
            targetType = [t for t in typeInfo if t.__name__ == label]
            if 0 == len(targetType):
                raise ValueError(f'modal type: {label} not found. supported modal type list: {[str(t.__name__) for t in typeInfo]}. please check your input.')
            else:
                return {'type': typeInfo[targetType[0]]['modal'], 'tag': m, 'content': targetType[0](param).Standardize()}
        else:
            mimeType = GuessMediaType(param)
            if 'image' in mimeType:
                return {'type': 'image', 'tag': m, 'content': AImageLocation(urlOrPath=param).Standardize(self.proxy)}
            elif 'video' in mimeType:
                return {'type': 'video', 'tag': m, 'content': AVideoLocation(urlOrPath=param).Standardize(self.proxy)}
            return

    def LatestEntry(self):
        for i in range(len(self.conversations)):
            if self.conversations[-i - 1]['entry']:
                break
        return -(i + 1) // 2 if 'ASSISTANT' == self.conversations[-1]['role'] else (-i - 2) // 2

    def GetConversations(self, frm=0):
        s = 2 * frm if frm >= 0 or 'ASSISTANT' == self.conversations[-1]['role'] else 2 * frm + 1
        return self.conversations[s:]

    def __len__(self):
        return (len(self.conversations) + 1) // 2

    def FromJson(self, data):

        def AddRecord(role, time, entry, msg, attachments):
            self.conversations.append({'role': role, 'time': time, 'entry': entry, 'msg': msg, 'attachments': attachments})
        for i in range(0, len(data)):
            d = data[i]
            if i > 0:
                assert not {d['role'], data[i - 1]['role']} <= {'ASSISTANT'}, f'Consecutive ASSISTANT messages were found in conversations. {str(d)}, {str(data[i - 1])}'
                if {d['role'], data[i - 1]['role']} <= {'USER', 'SYSTEM'}:
                    AddRecord('ASSISTANT', None, False, '<EMPTY MSG>', [])
            AddRecord(d['role'], d.get('time', None), d.get('entry', None), d['msg'] if '' != d['msg'] else '<EMPTY MSG>', [{'type': a['type'], 'tag': a.get('tag', None), 'content': FromJson(a['content'])} for a in d['attachments']])
        if len(data) > 0 and data[-1]['role'] in ['USER', 'SYSTEM']:
            AddRecord('ASSISTANT', None, False, '<EMPTY MSG>', [])
        return

    def ToJson(self) -> str:
        return [{'role': record['role'], 'time': record['time'], 'entry': record['entry'], 'msg': record['msg'], 'attachments': [{'type': a['type'], 'tag': a['tag'], 'content': ToJson(a['content'])} for a in record['attachments']]} for record in self.conversations]

def GuessMediaType(pathOrUrl: str) -> str:
    mimetype, _ = mimetypes.guess_type(pathOrUrl)
    if None != mimetype:
        return mimetype
    r = requests.head(pathOrUrl)
    return r.headers.get('content-type')

class AFormatterGPTVision:

    def __init__(self, tokenizer=None, systemAsUser=False):
        self.systemAsUser = systemAsUser
        return

    def ProcessAttachements(self, a):
        if 'image' == a['type']:
            return [{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{a['content'].ToJson()['data']}'}}]
        elif 'video' == a['type']:
            numFrames = 10
            video = av.open(io.BytesIO(a['content'].data))
            frameIndices = [int(i * video.streams.video[0].frames / (numFrames - 1)) for i in range(numFrames)]
            ret = []
            for index in frameIndices:
                video.seek(index)
                frame = next(video.decode(video=0)).to_image()
                bytesDst = io.BytesIO()
                frame.save(bytesDst, format='JPEG')
                image = AImage(data=bytesDst.getvalue())
                ret.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image.Standardize().ToJson()['data']}'}})
            video.close()
            return ret

    def BuildMsg(self, role: str, msg: str, attachments: list):
        roleMap = {'SYSTEM': 'system' if not self.systemAsUser else 'user', 'USER': 'user', 'ASSISTANT': 'assistant'}
        return {'role': roleMap[role], 'content': [{'type': 'text', 'text': msg}] + sum([self.ProcessAttachements(a) for a in attachments if a['type'] in ['image', 'video']], [])}

    def __call__(self, prompt0, conversations, encode=True, assistTag=True):
        ret = [{'role': 'system', 'content': [{'type': 'text', 'text': prompt0}]}] + [self.BuildMsg(c['role'], c['msg'], c['attachments']) for c in conversations]
        return (ret, TokenEstimatorOAI(conversations))


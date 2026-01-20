# Cluster 21

class AVideo(BaseModel):
    data: Optional[bytes]
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None

    def __init__(self, **params):
        super().__init__(**params)
        if self.data and (not all([self.format, self.width, self.height, self.fps])):
            meta = self.GetMeta()
            self.format = meta['format']
            self.width = meta['width']
            self.height = meta['height']
            self.fps = meta['fps']
        return

    def GetMeta(self):
        ret = {'width': 0, 'height': 0, 'fps': 0, 'format': None}
        if self.data:
            video = av.open(io.BytesIO(self.data))
            stream = next((s for s in video.streams if s.type == 'video'), None)
            if stream is not None:
                ret = {'width': stream.codec_context.width, 'height': stream.codec_context.height, 'fps': stream.average_rate, 'format': video.format}
            video.close()
        return ret

    def __str__(self) -> str:
        return f'< AVideo object in {self.format} format. >'

    @classmethod
    def FromJson(cls, data):
        return cls(data=base64.b64decode(data['data'].encode('utf-8')))

    def ToJson(self):
        return {'type': 'AVideo', 'format': self.format, 'data': base64.b64encode(self.data).decode('utf-8') if self.data else self.data}

    def Standardize(self):
        return AVideo(data=ConvertVideoFormat(self.data, 'mp4')) if self.data else self

def ConvertVideoFormat(bytesSrc, format):
    videoSrc = av.open(io.BytesIO(bytesSrc))
    if format == videoSrc.format:
        return bytesSrc.getvalue()
    bytesDst = io.BytesIO()
    videoDst = av.open(bytesDst, mode='w', format=format)
    streamSrc = videoSrc.streams.video[0]
    streamDst = videoDst.add_stream('libx264', rate=streamSrc.average_rate)
    streamDst.width = streamSrc.width
    streamDst.height = streamSrc.height
    streamDst.pix_fmt = 'yuv420p'
    for frame in videoSrc.decode(video=0):
        for packet in streamDst.encode(frame):
            videoDst.mux(packet)
    for packet in streamDst.encode():
        videoDst.mux(packet)
    videoSrc.close()
    videoDst.close()
    bytesDst.seek(0)
    return bytesDst.getvalue()

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


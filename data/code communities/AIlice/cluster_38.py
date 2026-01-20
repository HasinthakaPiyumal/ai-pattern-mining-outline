# Cluster 38

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

def TokenEstimatorOAI(conversations) -> int:
    ret = 0
    for c in conversations:
        ret += 4
        ret += len(c['msg']) // 4
        for a in c['attachments']:
            if 'image' == a['type']:
                ret += EstimateImageTokens(a['content'].width, a['content'].height)
            elif 'video' == a['type']:
                ret += EstimateImageTokens(a['content'].width, a['content'].height) * 10
    return ret

class AFormatterClaudeVision:

    def __init__(self, tokenizer=None, systemAsUser=False):
        self.systemAsUser = systemAsUser
        return

    def ProcessAttachements(self, a):
        if 'image' == a['type']:
            return [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': a['content'].ToJson()['data']}}]

    def BuildMsg(self, role: str, msg: str, attachments: list):
        roleMap = {'SYSTEM': 'system' if not self.systemAsUser else 'user', 'USER': 'user', 'ASSISTANT': 'assistant'}
        return {'role': roleMap[role], 'content': [{'type': 'text', 'text': msg}] + sum([self.ProcessAttachements(a) for a in attachments if a['type'] in ['image']], [])}

    def __call__(self, prompt0, conversations, encode=True, assistTag=True):
        ret = [{'role': 'system', 'content': [{'type': 'text', 'text': prompt0}]}] + [self.BuildMsg(c['role'], c['msg'], c['attachments']) for c in conversations]
        return (ret, TokenEstimatorOAI(conversations))


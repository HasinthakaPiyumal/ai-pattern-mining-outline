# Cluster 28

class ASpeech:

    def __init__(self):
        self.textQue = queue.Queue(maxsize=100)
        self.audioQue = queue.Queue(maxsize=100)
        self.inputDone = True
        self.lock = threading.Lock()
        self.noTextLeft = True
        self.textProcessor = threading.Thread(target=self.ProcessText, daemon=True)
        self.textProcessor.start()
        self.audioProcessor = threading.Thread(target=self.ProcessAudio, daemon=True)
        self.audioProcessor.start()
        return

    def ModuleInfo(self):
        return {'NAME': 'speech', 'ACTIONS': {'SPEECH-TO-TEXT': {'func': 'Speech2Text', 'prompt': 'Speech to text.', 'type': 'primary'}, 'TEXT-TO-SPEECH': {'func': 'Text2Speech', 'prompt': 'Text to speech.', 'type': 'primary'}, 'GET-AUDIO': {'func': 'GetAudio', 'prompt': 'Get text input from microphone.', 'type': 'primary'}, 'SPEAK': {'func': 'Speak', 'prompt': 'Synthesize input text fragments into audio and play.', 'type': 'primary'}, 'SWITCH-TONE': {'func': 'SwitchTone', 'prompt': 'Switch the TTS system to a new tone.', 'type': 'primary'}}}

    def PrepareModel(self):
        global s2t, t2s
        if None in [t2s, s2t]:
            t2s = T2S_ChatTTS()
            s2t = S2T_WhisperLarge()
        return

    def SetDevices(self, deviceMap: dict[str, str]):
        global s2t, t2s
        if 'stt' in deviceMap:
            s2t.To(deviceMap['stt'])
        elif 'tts' in deviceMap:
            t2s.To(deviceMap['tts'])
        return

    def Speech2Text(self, wav: list, sr: int) -> str:
        global s2t
        return s2t.recognize(audio_data_to_numpy((np.array(wav), sr)))

    def Text2Speech(self, txt: str) -> tuple[list, int]:
        global t2s
        if None == txt or '' == strip(txt):
            return ([1], 24000)
        audio, sr = t2s(txt)
        return (audio.tolist(), sr)

    def GetAudio(self) -> str:
        global s2t
        self.inputDone = True
        with self.lock:
            ret = s2t()
        return ret

    def Speak(self, txt: str):
        print('Speak(): ', txt)
        if None == txt or '' == strip(txt):
            return
        self.textQue.put(txt)
        self.inputDone = False
        return

    def SwitchTone(self) -> str:
        global t2s
        return t2s.SwitchTone()

    def ProcessText(self):
        global t2s
        while True:
            self.noTextLeft = self.inputDone and self.textQue.empty()
            text = self.textQue.get()
            try:
                self.audioQue.put(t2s(text))
            except Exception as e:
                print('EXCEPTION in ProcessText(). continue. e: ', str(e))
                continue

    def ProcessAudio(self):
        while True:
            time.sleep(0.1)
            with self.lock:
                while not (self.inputDone and self.noTextLeft and self.audioQue.empty()):
                    audio, sr = self.audioQue.get()
                    sd.play(audio, sr)
                    sd.wait()

def audio_data_to_numpy(audio_data, sr=16000):
    audio_array, sr0 = audio_data
    scale = np.iinfo(audio_array.dtype).max if audio_array.dtype in [np.int16, np.int32] else 1.0
    ret = librosa.resample(y=audio_array.astype(np.float32) / scale, orig_sr=sr0, target_sr=sr)
    return ret

class S2T_WhisperLarge:

    def __init__(self):
        self.device = 'cpu'
        self.processor = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
        self.model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3').to(self.device)
        self.model.config.forced_decoder_ids = None
        self.audio = AudioSourceSileroVAD()
        return

    def To(self, device: str):
        self.model = self.model.to(device)
        self.device = device
        return

    def __call__(self):
        said = ''
        while said == '':
            print('listening...')
            audio, sr = self.audio.get()
            print('got audio. processing...')
            try:
                said = self.recognize(audio_data_to_numpy((audio, sr), sr=16000))
                print('audio recognized: ', said)
            except Exception as e:
                print('Exception: ' + str(e))
                continue
        return said

    def recognize(self, audio):
        input_features = self.processor(audio, sampling_rate=16000, return_tensors='pt').input_features
        predicted_ids = self.model.generate(input_features.to(self.device))
        transcription = self.processor.batch_decode(predicted_ids.cpu(), skip_special_tokens=True)
        return transcription[0]


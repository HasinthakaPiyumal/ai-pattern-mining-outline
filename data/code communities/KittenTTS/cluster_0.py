# Cluster 0

class KittenTTS_1_Onnx:

    def __init__(self, model_path='kitten_tts_nano_preview.onnx', voices_path='voices.npz'):
        """Initialize KittenTTS with model and voice data.
        
        Args:
            model_path: Path to the ONNX model file
            voices_path: Path to the voices NPZ file
        """
        self.model_path = model_path
        self.voices = np.load(voices_path)
        self.session = ort.InferenceSession(model_path)
        self.phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
        self.text_cleaner = TextCleaner()
        self.available_voices = ['expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f']

    def _prepare_inputs(self, text: str, voice: str, speed: float=1.0) -> dict:
        """Prepare ONNX model inputs from text and voice parameters."""
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")
        phonemes_list = self.phonemizer.phonemize([text])
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = ' '.join(phonemes)
        tokens = self.text_cleaner(phonemes)
        tokens.insert(0, 0)
        tokens.append(0)
        input_ids = np.array([tokens], dtype=np.int64)
        ref_s = self.voices[voice]
        return {'input_ids': input_ids, 'style': ref_s, 'speed': np.array([speed], dtype=np.float32)}

    def generate(self, text: str, voice: str='expr-voice-5-m', speed: float=1.0) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            
        Returns:
            Audio data as numpy array
        """
        onnx_inputs = self._prepare_inputs(text, voice, speed)
        outputs = self.session.run(None, onnx_inputs)
        audio = outputs[0][5000:-10000]
        return audio

    def generate_to_file(self, text: str, output_path: str, voice: str='expr-voice-5-m', speed: float=1.0, sample_rate: int=24000) -> None:
        """Synthesize speech and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
        """
        audio = self.generate(text, voice, speed)
        sf.write(output_path, audio, sample_rate)
        print(f'Audio saved to {output_path}')


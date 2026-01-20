# Cluster 3

def ensure_lasinya_models(models_root: str='models', model_name: str='Lasinya') -> None:
    """
    Ensures the Coqui XTTS Lasinya model files are present locally.

    Checks for required model files (config.json, vocab.json, etc.) within
    the specified directory structure. If any file is missing, it downloads
    it from the 'KoljaB/XTTS_Lasinya' Hugging Face Hub repository.

    Args:
        models_root: The root directory where models are stored.
        model_name: The specific name of the model subdirectory.
    """
    base = os.path.join(models_root, model_name)
    create_directory(base)
    files = ['config.json', 'vocab.json', 'speakers_xtts.pth', 'model.pth']
    for fn in files:
        local_file = os.path.join(base, fn)
        if not os.path.exists(local_file):
            print(f'👄⏬ Downloading {fn} to {base}')
            hf_hub_download(repo_id='KoljaB/XTTS_Lasinya', filename=fn, local_dir=base)

def create_directory(path: str) -> None:
    """
    Creates a directory at the specified path if it doesn't already exist.

    Args:
        path: The directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

class AudioProcessor:
    """
    Manages Text-to-Speech (TTS) synthesis using various engines via RealtimeTTS.

    This class initializes a chosen TTS engine (Coqui, Kokoro, or Orpheus),
    configures it for streaming output, measures initial latency (TTFT),
    and provides methods to synthesize audio from text strings or generators,
    placing the resulting audio chunks into a queue. It handles dynamic
    stream parameter adjustments and manages the synthesis lifecycle, including
    optional callbacks upon receiving the first audio chunk.
    """

    def __init__(self, engine: str=START_ENGINE, orpheus_model: str='orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf') -> None:
        """
        Initializes the AudioProcessor with a specific TTS engine.

        Sets up the chosen engine (Coqui, Kokoro, Orpheus), downloads Coqui models
        if necessary, configures the RealtimeTTS stream, and performs an initial
        synthesis to measure Time To First Audio chunk (TTFA).

        Args:
            engine: The name of the TTS engine to use ("coqui", "kokoro", "orpheus").
            orpheus_model: The path or identifier for the Orpheus model file (used only if engine is "orpheus").
        """
        self.engine_name = engine
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.audio_chunks = asyncio.Queue()
        self.orpheus_model = orpheus_model
        self.silence = ENGINE_SILENCES.get(engine, ENGINE_SILENCES[self.engine_name])
        self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE
        if engine == 'coqui':
            ensure_lasinya_models(models_root='models', model_name='Lasinya')
            self.engine = CoquiEngine(specific_model='Lasinya', local_models_path='./models', voice='reference_audio.wav', speed=1.1, use_deepspeed=True, thread_count=6, stream_chunk_size=self.current_stream_chunk_size, overlap_wav_len=1024, load_balancing=True, load_balancing_buffer_length=0.5, load_balancing_cut_off=0.1, add_sentence_filter=True)
        elif engine == 'kokoro':
            self.engine = KokoroEngine(voice='af_heart', default_speed=1.26, trim_silence=True, silence_threshold=0.01, extra_start_ms=25, extra_end_ms=15, fade_in_ms=15, fade_out_ms=10)
        elif engine == 'orpheus':
            self.engine = OrpheusEngine(model=self.orpheus_model, temperature=0.8, top_p=0.95, repetition_penalty=1.1, max_tokens=1200)
            voice = OrpheusVoice('tara')
            self.engine.set_voice(voice)
        else:
            raise ValueError(f'Unsupported engine: {engine}')
        self.stream = TextToAudioStream(self.engine, muted=True, playout_chunk_size=4096, on_audio_stream_stop=self.on_audio_stream_stop)
        if self.engine_name == 'coqui' and hasattr(self.engine, 'set_stream_chunk_size') and (self.current_stream_chunk_size != QUICK_ANSWER_STREAM_CHUNK_SIZE):
            logger.info(f'👄⚙️ Setting Coqui stream chunk size to {QUICK_ANSWER_STREAM_CHUNK_SIZE} for initial setup.')
            self.engine.set_stream_chunk_size(QUICK_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE
        self.stream.feed('prewarm')
        play_kwargs = dict(log_synthesized_text=False, muted=True, fast_sentence_fragment=False, comma_silence_duration=self.silence.comma, sentence_silence_duration=self.silence.sentence, default_silence_duration=self.silence.default, force_first_fragment_after_words=999999)
        self.stream.play(**play_kwargs)
        while self.stream.is_playing():
            time.sleep(0.01)
        self.finished_event.wait()
        self.finished_event.clear()
        start_time = time.time()
        ttfa = None

        def on_audio_chunk_ttfa(chunk: bytes):
            nonlocal ttfa
            if ttfa is None:
                ttfa = time.time() - start_time
                logger.debug(f'👄⏱️ TTFA measurement first chunk arrived, TTFA: {ttfa:.2f}s.')
        self.stream.feed('This is a test sentence to measure the time to first audio chunk.')
        play_kwargs_ttfa = dict(on_audio_chunk=on_audio_chunk_ttfa, log_synthesized_text=False, muted=True, fast_sentence_fragment=False, comma_silence_duration=self.silence.comma, sentence_silence_duration=self.silence.sentence, default_silence_duration=self.silence.default, force_first_fragment_after_words=999999)
        self.stream.play_async(**play_kwargs_ttfa)
        while ttfa is None and (self.stream.is_playing() or not self.finished_event.is_set()):
            time.sleep(0.01)
        self.stream.stop()
        if not self.finished_event.is_set():
            self.finished_event.wait(timeout=2.0)
        self.finished_event.clear()
        if ttfa is not None:
            logger.debug(f'👄⏱️ TTFA measurement complete. TTFA: {ttfa:.2f}s.')
            self.tts_inference_time = ttfa * 1000
        else:
            logger.warning('👄⚠️ TTFA measurement failed (no audio chunk received).')
            self.tts_inference_time = 0
        self.on_first_audio_chunk_synthesize: Optional[Callable[[], None]] = None

    def on_audio_stream_stop(self) -> None:
        """
        Callback executed when the RealtimeTTS audio stream stops processing.

        Logs the event and sets the `finished_event` to signal completion or stop.
        """
        logger.info('👄🛑 Audio stream stopped.')
        self.finished_event.set()

    def synthesize(self, text: str, audio_chunks: Queue, stop_event: threading.Event, generation_string: str='') -> bool:
        """
        Synthesizes audio from a complete text string and puts chunks into a queue.

        Feeds the entire text string to the TTS engine. As audio chunks are generated,
        they are potentially buffered initially for smoother streaming and then put
        into the provided queue. Synthesis can be interrupted via the stop_event.
        Skips initial silent chunks if using the Orpheus engine. Triggers the
        `on_first_audio_chunk_synthesize` callback when the first valid audio chunk is queued.

        Args:
            text: The text string to synthesize.
            audio_chunks: The queue to put the resulting audio chunks (bytes) into.
                          This should typically be the instance's `self.audio_chunks`.
            stop_event: A threading.Event to signal interruption of the synthesis.
                        This should typically be the instance's `self.stop_event`.
            generation_string: An optional identifier string for logging purposes.

        Returns:
            True if synthesis completed fully, False if interrupted by stop_event.
        """
        if self.engine_name == 'coqui' and hasattr(self.engine, 'set_stream_chunk_size') and (self.current_stream_chunk_size != QUICK_ANSWER_STREAM_CHUNK_SIZE):
            logger.info(f'👄⚙️ {generation_string} Setting Coqui stream chunk size to {QUICK_ANSWER_STREAM_CHUNK_SIZE} for quick synthesis.')
            self.engine.set_stream_chunk_size(QUICK_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE
        self.stream.feed(text)
        self.finished_event.clear()
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = (24000, 2)
        start = time.time()
        self._quick_prev_chunk_time: float = 0.0

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            if stop_event.is_set():
                logger.info(f'👄🛑 {generation_string} Quick audio stream interrupted by stop_event. Text: {text[:50]}...')
                return
            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR
            if on_audio_chunk.first_call and self.engine_name == 'orpheus':
                if not hasattr(on_audio_chunk, 'silent_chunks_count'):
                    on_audio_chunk.silent_chunks_count = 0
                    on_audio_chunk.silent_chunks_time = 0.0
                    on_audio_chunk.silence_threshold = 200
                try:
                    fmt = f'{samples}h'
                    pcm_data = struct.unpack(fmt, chunk)
                    avg_amplitude = np.abs(np.array(pcm_data)).mean()
                    if avg_amplitude < on_audio_chunk.silence_threshold:
                        on_audio_chunk.silent_chunks_count += 1
                        on_audio_chunk.silent_chunks_time += play_duration
                        logger.debug(f'👄⏭️ {generation_string} Quick Skipping silent chunk {on_audio_chunk.silent_chunks_count} (avg_amp: {avg_amplitude:.2f})')
                        return
                    elif on_audio_chunk.silent_chunks_count > 0:
                        logger.info(f'👄⏭️ {generation_string} Quick Skipped {on_audio_chunk.silent_chunks_count} silent chunks, saved {on_audio_chunk.silent_chunks_time * 1000:.2f}ms')
                except Exception as e:
                    logger.warning(f'👄⚠️ {generation_string} Quick Error analyzing audio chunk for silence: {e}')
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._quick_prev_chunk_time = now
                ttfa_actual = now - start
                logger.info(f'👄🚀 {generation_string} Quick audio start. TTFA: {ttfa_actual:.2f}s. Text: {text[:50]}...')
            else:
                gap = now - self._quick_prev_chunk_time
                self._quick_prev_chunk_time = now
                if gap <= play_duration * 1.1:
                    good_streak += 1
                else:
                    logger.warning(f'👄❌ {generation_string} Quick chunk slow (gap={gap:.3f}s > {play_duration:.3f}s). Text: {text[:50]}...')
                    good_streak = 0
            put_occurred_this_call = False
            buffer.append(chunk)
            buf_dur += play_duration
            if buffering:
                if good_streak >= 2 or buf_dur >= 0.5:
                    logger.info(f'👄➡️ {generation_string} Quick Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s).')
                    for c in buffer:
                        try:
                            audio_chunks.put_nowait(c)
                            put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f'👄⚠️ {generation_string} Quick audio queue full, dropping chunk.')
                    buffer.clear()
                    buf_dur = 0.0
                    buffering = False
            else:
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f'👄⚠️ {generation_string} Quick audio queue full, dropping chunk.')
            if put_occurred_this_call and (not on_audio_chunk.callback_fired):
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f'👄🚀 {generation_string} Quick Firing on_first_audio_chunk_synthesize.')
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(f'👄💥 {generation_string} Quick Error in on_first_audio_chunk_synthesize callback: {e}', exc_info=True)
                on_audio_chunk.callback_fired = True
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False
        play_kwargs = dict(log_synthesized_text=True, on_audio_chunk=on_audio_chunk, muted=True, fast_sentence_fragment=False, comma_silence_duration=self.silence.comma, sentence_silence_duration=self.silence.sentence, default_silence_duration=self.silence.default, force_first_fragment_after_words=999999)
        logger.info(f'👄▶️ {generation_string} Quick Starting synthesis. Text: {text[:50]}...')
        self.stream.play_async(**play_kwargs)
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f'👄🛑 {generation_string} Quick answer synthesis aborted by stop_event. Text: {text[:50]}...')
                buffer.clear()
                self.finished_event.wait(timeout=1.0)
                return False
            time.sleep(0.01)
        if buffering and buffer and (not stop_event.is_set()):
            logger.info(f'👄➡️ {generation_string} Quick Flushing remaining buffer after stream finished.')
            for c in buffer:
                try:
                    audio_chunks.put_nowait(c)
                except asyncio.QueueFull:
                    logger.warning(f'👄⚠️ {generation_string} Quick audio queue full on final flush, dropping chunk.')
            buffer.clear()
        logger.info(f'👄✅ {generation_string} Quick answer synthesis complete. Text: {text[:50]}...')
        return True

    def synthesize_generator(self, generator: Generator[str, None, None], audio_chunks: Queue, stop_event: threading.Event, generation_string: str='') -> bool:
        """
        Synthesizes audio from a generator yielding text chunks and puts audio into a queue.

        Feeds text chunks yielded by the generator to the TTS engine. As audio chunks
        are generated, they are potentially buffered initially and then put into the
        provided queue. Synthesis can be interrupted via the stop_event.
        Skips initial silent chunks if using the Orpheus engine. Sets specific playback
        parameters when using the Orpheus engine. Triggers the
       `on_first_audio_chunk_synthesize` callback when the first valid audio chunk is queued.


        Args:
            generator: A generator yielding text chunks (strings) to synthesize.
            audio_chunks: The queue to put the resulting audio chunks (bytes) into.
                          This should typically be the instance's `self.audio_chunks`.
            stop_event: A threading.Event to signal interruption of the synthesis.
                        This should typically be the instance's `self.stop_event`.
            generation_string: An optional identifier string for logging purposes.

        Returns:
            True if synthesis completed fully, False if interrupted by stop_event.
        """
        if self.engine_name == 'coqui' and hasattr(self.engine, 'set_stream_chunk_size') and (self.current_stream_chunk_size != FINAL_ANSWER_STREAM_CHUNK_SIZE):
            logger.info(f'👄⚙️ {generation_string} Setting Coqui stream chunk size to {FINAL_ANSWER_STREAM_CHUNK_SIZE} for generator synthesis.')
            self.engine.set_stream_chunk_size(FINAL_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = FINAL_ANSWER_STREAM_CHUNK_SIZE
        self.stream.feed(generator)
        self.finished_event.clear()
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = (24000, 2)
        start = time.time()
        self._final_prev_chunk_time: float = 0.0

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            if stop_event.is_set():
                logger.info(f'👄🛑 {generation_string} Final audio stream interrupted by stop_event.')
                return
            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR
            if on_audio_chunk.first_call and self.engine_name == 'orpheus':
                if not hasattr(on_audio_chunk, 'silent_chunks_count'):
                    on_audio_chunk.silent_chunks_count = 0
                    on_audio_chunk.silent_chunks_time = 0.0
                    on_audio_chunk.silence_threshold = 100
                try:
                    fmt = f'{samples}h'
                    pcm_data = struct.unpack(fmt, chunk)
                    avg_amplitude = np.abs(np.array(pcm_data)).mean()
                    if avg_amplitude < on_audio_chunk.silence_threshold:
                        on_audio_chunk.silent_chunks_count += 1
                        on_audio_chunk.silent_chunks_time += play_duration
                        logger.debug(f'👄⏭️ {generation_string} Final Skipping silent chunk {on_audio_chunk.silent_chunks_count} (avg_amp: {avg_amplitude:.2f})')
                        return
                    elif on_audio_chunk.silent_chunks_count > 0:
                        logger.info(f'👄⏭️ {generation_string} Final Skipped {on_audio_chunk.silent_chunks_count} silent chunks, saved {on_audio_chunk.silent_chunks_time * 1000:.2f}ms')
                except Exception as e:
                    logger.warning(f'👄⚠️ {generation_string} Final Error analyzing audio chunk for silence: {e}')
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._final_prev_chunk_time = now
                ttfa_actual = now - start
                logger.info(f'👄🚀 {generation_string} Final audio start. TTFA: {ttfa_actual:.2f}s.')
            else:
                gap = now - self._final_prev_chunk_time
                self._final_prev_chunk_time = now
                if gap <= play_duration * 1.1:
                    good_streak += 1
                else:
                    logger.warning(f'👄❌ {generation_string} Final chunk slow (gap={gap:.3f}s > {play_duration:.3f}s).')
                    good_streak = 0
            put_occurred_this_call = False
            buffer.append(chunk)
            buf_dur += play_duration
            if buffering:
                if good_streak >= 2 or buf_dur >= 0.5:
                    logger.info(f'👄➡️ {generation_string} Final Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s).')
                    for c in buffer:
                        try:
                            audio_chunks.put_nowait(c)
                            put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f'👄⚠️ {generation_string} Final audio queue full, dropping chunk.')
                    buffer.clear()
                    buf_dur = 0.0
                    buffering = False
            else:
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f'👄⚠️ {generation_string} Final audio queue full, dropping chunk.')
            if put_occurred_this_call and (not on_audio_chunk.callback_fired):
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f'👄🚀 {generation_string} Final Firing on_first_audio_chunk_synthesize.')
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(f'👄💥 {generation_string} Final Error in on_first_audio_chunk_synthesize callback: {e}', exc_info=True)
                on_audio_chunk.callback_fired = True
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False
        play_kwargs = dict(log_synthesized_text=True, on_audio_chunk=on_audio_chunk, muted=True, fast_sentence_fragment=False, comma_silence_duration=self.silence.comma, sentence_silence_duration=self.silence.sentence, default_silence_duration=self.silence.default, force_first_fragment_after_words=999999)
        if self.engine_name == 'orpheus':
            play_kwargs['minimum_sentence_length'] = 200
            play_kwargs['minimum_first_fragment_length'] = 200
        logger.info(f'👄▶️ {generation_string} Final Starting synthesis from generator.')
        self.stream.play_async(**play_kwargs)
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f'👄🛑 {generation_string} Final answer synthesis aborted by stop_event.')
                buffer.clear()
                self.finished_event.wait(timeout=1.0)
                return False
            time.sleep(0.01)
        if buffering and buffer and (not stop_event.is_set()):
            logger.info(f'👄➡️ {generation_string} Final Flushing remaining buffer after stream finished.')
            for c in buffer:
                try:
                    audio_chunks.put_nowait(c)
                except asyncio.QueueFull:
                    logger.warning(f'👄⚠️ {generation_string} Final audio queue full on final flush, dropping chunk.')
            buffer.clear()
        logger.info(f'👄✅ {generation_string} Final answer synthesis complete.')
        return True


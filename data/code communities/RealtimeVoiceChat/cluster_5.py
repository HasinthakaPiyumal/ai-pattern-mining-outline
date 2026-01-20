# Cluster 5

def strip_ending_punctuation(text: str) -> str:
    """
    Removes trailing punctuation marks defined in `sentence_end_marks`.

    Removes trailing whitespace first, then iteratively removes any characters
    from `sentence_end_marks` found at the end of the string.

    Args:
        text: The input text string.

    Returns:
        The text string with specified trailing punctuation removed.
    """
    text = text.rstrip()
    for char in sentence_end_marks:
        while text.endswith(char):
            text = text.rstrip(char)
    return text

class TranscriptionProcessor:
    """
    Manages audio transcription using RealtimeSTT, handling real-time and final
    transcription callbacks, silence detection, turn detection (optional),
    and potential sentence end detection.

    This class acts as a bridge between raw audio input and transcription results,
    coordinating the RealtimeSTT recorder, processing callbacks, and managing
    internal state related to silence, potential sentences, and turn timing.
    """
    _PIPELINE_RESERVE_TIME_MS: float = 0.02
    _HOT_THRESHOLD_OFFSET_S: float = 0.35
    _MIN_HOT_CONDITION_DURATION_S: float = 0.15
    _TTS_ALLOWANCE_OFFSET_S: float = 0.25
    _MIN_POTENTIAL_END_DETECTION_TIME_MS: float = 0.02
    _SENTENCE_CACHE_MAX_AGE_MS: float = 0.2
    _SENTENCE_CACHE_TRIGGER_COUNT: int = 3

    def __init__(self, source_language: str='en', realtime_transcription_callback: Optional[Callable[[str], None]]=None, full_transcription_callback: Optional[Callable[[str], None]]=None, potential_full_transcription_callback: Optional[Callable[[str], None]]=None, potential_full_transcription_abort_callback: Optional[Callable[[], None]]=None, potential_sentence_end: Optional[Callable[[str], None]]=None, before_final_sentence: Optional[Callable[[Optional[np.ndarray], Optional[str]], bool]]=None, silence_active_callback: Optional[Callable[[bool], None]]=None, on_recording_start_callback: Optional[Callable[[], None]]=None, is_orpheus: bool=False, local: bool=True, tts_allowed_event: Optional[threading.Event]=None, pipeline_latency: float=0.5, recorder_config: Optional[Dict[str, Any]]=None) -> None:
        """
        Initializes the TranscriptionProcessor.

        Args:
            source_language: Language code for transcription (e.g., "en").
            realtime_transcription_callback: Callback for real-time transcription updates. Receives partial text.
            full_transcription_callback: Callback for final transcription result. Receives final text.
            potential_full_transcription_callback: Callback triggered when a full transcription is likely imminent (in "hot" state). Receives current real-time text.
            potential_full_transcription_abort_callback: Callback triggered when the "hot" state ends before final transcription.
            potential_sentence_end: Callback triggered when a potential sentence end is detected. Receives the potentially complete sentence text.
            before_final_sentence: Callback triggered just before the recorder finalizes transcription. Receives audio copy and current real-time text. Return True to potentially influence recorder behavior (if supported).
            silence_active_callback: Callback triggered when silence detection state changes. Receives boolean (True if silence is active).
            on_recording_start_callback: Callback triggered when the recorder starts recording after silence or wake word.
            is_orpheus: Flag indicating if specific timing adjustments for 'Orpheus' mode should be used.
            local: Flag used by TurnDetection (if enabled) to indicate local vs remote processing context.
            tts_allowed_event: An event that might be set when TTS synthesis is allowed (currently unused in provided logic).
            pipeline_latency: Estimated latency of the downstream processing pipeline in seconds. Used for timing calculations.
            recorder_config: Optional dictionary to override default RealtimeSTT recorder configuration.
        """
        self.source_language = source_language
        self.realtime_transcription_callback = realtime_transcription_callback
        self.full_transcription_callback = full_transcription_callback
        self.potential_full_transcription_callback = potential_full_transcription_callback
        self.potential_full_transcription_abort_callback = potential_full_transcription_abort_callback
        self.potential_sentence_end = potential_sentence_end
        self.before_final_sentence = before_final_sentence
        self.silence_active_callback = silence_active_callback
        self.on_recording_start_callback = on_recording_start_callback
        self.is_orpheus = is_orpheus
        self.pipeline_latency = pipeline_latency
        self.recorder: Optional[AudioToTextRecorder | AudioToTextRecorderClient] = None
        self.is_silero_speech_active: bool = False
        self.silero_working: bool = False
        self.on_wakeword_detection_start: Optional[Callable] = None
        self.on_wakeword_detection_end: Optional[Callable] = None
        self.realtime_text: Optional[str] = None
        self.sentence_end_cache: List[Dict[str, Any]] = []
        self.potential_sentences_yielded: List[Dict[str, Any]] = []
        self.stripped_partial_user_text: str = ''
        self.final_transcription: Optional[str] = None
        self.shutdown_performed: bool = False
        self.silence_time: float = 0.0
        self.silence_active: bool = False
        self.last_audio_copy: Optional[np.ndarray] = None
        self.on_tts_allowed_to_synthesize: Optional[Callable] = None
        self.text_similarity = TextSimilarity(focus='end', n_words=5)
        self.recorder_config = copy.deepcopy(recorder_config if recorder_config else DEFAULT_RECORDER_CONFIG)
        self.recorder_config['language'] = self.source_language
        if USE_TURN_DETECTION:
            logger.info(f'👂🔄 {Colors.YELLOW}Turn detection enabled{Colors.RESET}')
            self.turn_detection = TurnDetection(on_new_waiting_time=self.on_new_waiting_time, local=local, pipeline_latency=pipeline_latency)
        self._create_recorder()
        self._start_silence_monitor()

    def _get_recorder_param(self, param_name: str, default: Any=None) -> Any:
        """
        Internal helper to get a parameter from the recorder instance,
        abstracting client/server differences.

        Args:
            param_name: The name of the parameter to retrieve.
            default: The value to return if the recorder is not initialized or
                     the parameter doesn't exist.

        Returns:
            The value of the recorder parameter or the default value.
        """
        if not self.recorder:
            return default
        if START_STT_SERVER:
            return self.recorder.get_parameter(param_name)
        else:
            return getattr(self.recorder, param_name, default)

    def _set_recorder_param(self, param_name: str, value: Any) -> None:
        """
        Internal helper to set a parameter on the recorder instance,
        abstracting client/server differences.

        Args:
            param_name: The name of the parameter to set.
            value: The value to set the parameter to.
        """
        if not self.recorder:
            return
        if START_STT_SERVER:
            self.recorder.set_parameter(param_name, value)
        else:
            setattr(self.recorder, param_name, value)

    def _is_recorder_recording(self) -> bool:
        """
        Internal helper to check if the recorder is currently recording,
        abstracting client/server differences.

        Returns:
            True if the recorder is active and recording, False otherwise.
        """
        if not self.recorder:
            return False
        if START_STT_SERVER:
            return self.recorder.get_parameter('is_recording')
        else:
            return getattr(self.recorder, 'is_recording', False)

    def _start_silence_monitor(self) -> None:
        """
        Starts a background thread to monitor silence duration and trigger
        events like potential sentence end detection, TTS synthesis allowance,
        and potential full transcription ("hot") state changes.
        """

        def monitor():
            hot = False
            self.silence_time = self._get_recorder_param('speech_end_silence_start', 0.0)
            while not self.shutdown_performed:
                speech_end_silence_start = self.silence_time
                if self.recorder and speech_end_silence_start is not None and (speech_end_silence_start != 0):
                    silence_waiting_time = self._get_recorder_param('post_speech_silence_duration', 0.0)
                    time_since_silence = time.time() - speech_end_silence_start
                    latest_pipe_start_time = silence_waiting_time - self.pipeline_latency - self._PIPELINE_RESERVE_TIME_MS
                    potential_sentence_end_time = latest_pipe_start_time
                    if potential_sentence_end_time < self._MIN_POTENTIAL_END_DETECTION_TIME_MS:
                        potential_sentence_end_time = self._MIN_POTENTIAL_END_DETECTION_TIME_MS
                    start_hot_condition_time = silence_waiting_time - self._HOT_THRESHOLD_OFFSET_S
                    if start_hot_condition_time < self._MIN_HOT_CONDITION_DURATION_S:
                        start_hot_condition_time = self._MIN_HOT_CONDITION_DURATION_S
                    if self.is_orpheus:
                        orpheus_potential_end_time = silence_waiting_time - self._HOT_THRESHOLD_OFFSET_S
                        if potential_sentence_end_time < orpheus_potential_end_time:
                            potential_sentence_end_time = orpheus_potential_end_time
                    if time_since_silence > potential_sentence_end_time:
                        current_text = self.realtime_text if self.realtime_text else ''
                        logger.info(f'👂🔚 {Colors.YELLOW}Potential sentence end detected (timed out){Colors.RESET}: {current_text}')
                        self.detect_potential_sentence_end(current_text, force_yield=True, force_ellipses=True)
                    tts_allowance_time = silence_waiting_time - self._TTS_ALLOWANCE_OFFSET_S
                    if time_since_silence > tts_allowance_time:
                        if self.on_tts_allowed_to_synthesize:
                            self.on_tts_allowed_to_synthesize()
                    hot_condition_met = time_since_silence > start_hot_condition_time
                    if hot_condition_met and (not hot):
                        hot = True
                        print(f'{Colors.MAGENTA}HOT{Colors.RESET}')
                        if self.potential_full_transcription_callback:
                            self.potential_full_transcription_callback(self.realtime_text)
                    elif not hot_condition_met and hot:
                        if self._is_recorder_recording():
                            print(f'{Colors.CYAN}COLD (during silence){Colors.RESET}')
                            if self.potential_full_transcription_abort_callback:
                                self.potential_full_transcription_abort_callback()
                        hot = False
                elif hot:
                    if self._is_recorder_recording():
                        print(f'{Colors.CYAN}COLD (silence ended){Colors.RESET}')
                        if self.potential_full_transcription_abort_callback:
                            self.potential_full_transcription_abort_callback()
                    hot = False
                time.sleep(0.001)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def on_new_waiting_time(self, waiting_time: float, text: Optional[str]=None) -> None:
        """
        Callback handler for when TurnDetection calculates a new waiting time.
        Updates the recorder's post_speech_silence_duration parameter.

        Args:
            waiting_time: The new calculated silence duration in seconds.
            text: The text used by TurnDetection to calculate the waiting time (for logging).
        """
        if self.recorder:
            current_duration = self._get_recorder_param('post_speech_silence_duration')
            if current_duration != waiting_time:
                log_text = text if text else '(No text provided)'
                logger.info(f'👂⏳ {Colors.GRAY}New waiting time: {Colors.RESET}{Colors.YELLOW}{waiting_time:.2f}{Colors.RESET}{Colors.GRAY} for text: {log_text}{Colors.RESET}')
                self._set_recorder_param('post_speech_silence_duration', waiting_time)
        else:
            logger.warning('👂⚠️ Recorder not initialized, cannot set new waiting time.')

    def transcribe_loop(self) -> None:
        """
        Sets up the final transcription callback mechanism with the recorder.

        This method defines the `on_final` callback that will be invoked by the
        recorder when a complete utterance transcription is available. It then
        registers this callback with the recorder instance.
        """

        def on_final(text: Optional[str]):
            if text is None or text == '':
                logger.warning('👂❓ Final transcription received None or empty string.')
                return
            self.final_transcription = text
            logger.info(f'👂✅ {Colors.apply('Final user text: ').green} {Colors.apply(text).yellow}')
            self.sentence_end_cache.clear()
            self.potential_sentences_yielded.clear()
            if USE_TURN_DETECTION and hasattr(self, 'turn_detection'):
                self.turn_detection.reset()
            if self.full_transcription_callback:
                self.full_transcription_callback(text)
        if self.recorder:
            if hasattr(self.recorder, 'text'):
                self.recorder.text(on_final)
            elif START_STT_SERVER:
                logger.warning("👂⚠️ Recorder client does not have a 'text' method. Attempting to set 'on_final_transcription' parameter.")
                try:
                    self._set_recorder_param('on_final_transcription', on_final)
                except Exception as e:
                    logger.error(f'👂💥 Failed to set final transcription callback parameter for client: {e}')
            else:
                logger.warning("👂⚠️ Local recorder object does not have a 'text' method for final callback.")
        else:
            logger.error('👂❌ Cannot set final callback: Recorder not initialized.')

    def abort_generation(self) -> None:
        """
        Clears the cache of potentially yielded sentences.

        This effectively stops any further actions that might be triggered based
        on previously detected potential sentence ends, useful if processing needs
        to be reset or interrupted externally.
        """
        self.potential_sentences_yielded.clear()
        logger.info('👂⏹️ Potential sentence yield cache cleared (generation aborted).')

    def perform_final(self, audio_bytes: Optional[bytes]=None) -> None:
        """
        Manually triggers the final transcription process using the last known
        real-time text.

        This bypasses the recorder's natural end-of-speech detection and immediately
        invokes the final transcription callback with the `self.realtime_text` content.
        Useful for scenarios where transcription needs to be finalized externally.

        Args:
            audio_bytes: Optional audio data (currently unused in this method's logic
                         but kept for potential future use or API consistency).
        """
        if self.recorder:
            if self.realtime_text is None:
                logger.warning(f'👂❓ {Colors.RED}Forcing final transcription, but realtime_text is None. Using empty string.{Colors.RESET}')
                current_text = ''
            else:
                current_text = self.realtime_text
            self.final_transcription = current_text
            logger.info(f'👂❗ {Colors.apply('Forced Final user text: ').green} {Colors.apply(current_text).yellow}')
            self.sentence_end_cache.clear()
            self.potential_sentences_yielded.clear()
            if USE_TURN_DETECTION and hasattr(self, 'turn_detection'):
                self.turn_detection.reset()
            if self.full_transcription_callback:
                self.full_transcription_callback(current_text)
        else:
            logger.warning('👂⚠️ Cannot perform final: Recorder not initialized.')

    def _normalize_text(self, text: str) -> str:
        """
        Internal helper to normalize text for comparison purposes.
        Converts to lowercase, removes non-alphanumeric characters (except spaces),
        and collapses extra whitespace.

        Args:
            text: The input string to normalize.

        Returns:
            The normalized string.
        """
        text = text.lower()
        text = re.sub('[^a-z0-9\\s]', '', text)
        text = re.sub('\\s+', ' ', text).strip()
        return text

    def is_basically_the_same(self, text1: str, text2: str, similarity_threshold: float=0.96) -> bool:
        """
        Checks if two text strings are highly similar, focusing on the ending words.
        Uses the internal TextSimilarity instance.

        Args:
            text1: The first text string.
            text2: The second text string.
            similarity_threshold: The minimum similarity score (0 to 1) to consider
                                  the texts the same.

        Returns:
            True if the similarity score exceeds the threshold, False otherwise.
        """
        similarity = self.text_similarity.calculate_similarity(text1, text2)
        return similarity > similarity_threshold

    def detect_potential_sentence_end(self, text: Optional[str], force_yield: bool=False, force_ellipses: bool=False) -> None:
        """
        Detects potential sentence endings based on ending punctuation and timing stability.

        Checks if the provided text ends with sentence-ending punctuation (., !, ?).
        If so, it caches the normalized text and its timestamp. If the same normalized
        text ending appears frequently within a short time window or if `force_yield`
        is True (e.g., due to silence timeout), it triggers the `potential_sentence_end`
        callback, avoiding redundant triggers for the same sentence.

        Args:
            text: The real-time transcription text to check.
            force_yield: If True, bypasses punctuation and timing checks and forces
                         triggering the callback (if text is valid and not already yielded).
            force_ellipses: If True (used with `force_yield`), allows "..." to be
                            considered a sentence end.
        """
        if not text:
            return
        stripped_text_raw = text.strip()
        if not stripped_text_raw:
            return
        if stripped_text_raw.endswith('...') and (not force_ellipses):
            return
        end_punctuations = ['.', '!', '?']
        now = time.time()
        ends_with_punctuation = any((stripped_text_raw.endswith(p) for p in end_punctuations))
        if not ends_with_punctuation and (not force_yield):
            return
        normalized_text = self._normalize_text(stripped_text_raw)
        if not normalized_text:
            return
        entry_found = None
        for entry in self.sentence_end_cache:
            if self.is_basically_the_same(entry['text'], normalized_text):
                entry_found = entry
                break
        if entry_found:
            entry_found['timestamps'].append(now)
            entry_found['timestamps'] = [t for t in entry_found['timestamps'] if now - t <= self._SENTENCE_CACHE_MAX_AGE_MS]
        else:
            entry_found = {'text': normalized_text, 'timestamps': [now]}
            self.sentence_end_cache.append(entry_found)
        should_yield = False
        if force_yield:
            should_yield = True
        elif ends_with_punctuation and len(entry_found['timestamps']) >= self._SENTENCE_CACHE_TRIGGER_COUNT:
            should_yield = True
        if should_yield:
            already_yielded = False
            for yielded_entry in self.potential_sentences_yielded:
                if self.is_basically_the_same(yielded_entry['text'], normalized_text):
                    already_yielded = True
                    break
            if not already_yielded:
                self.potential_sentences_yielded.append({'text': normalized_text, 'timestamp': now})
                logger.info(f'👂➡️ Yielding potential sentence end: {stripped_text_raw}')
                if self.potential_sentence_end:
                    self.potential_sentence_end(stripped_text_raw)

    def set_silence(self, silence_active: bool) -> None:
        """
        Updates the internal silence state and triggers the silence_active_callback.

        Args:
            silence_active: The new silence state (True if silence is now active).
        """
        if self.silence_active != silence_active:
            self.silence_active = silence_active
            logger.info(f'👂🤫 Silence state changed: {('ACTIVE' if silence_active else 'INACTIVE')}')
            if self.silence_active_callback:
                self.silence_active_callback(silence_active)

    def get_last_audio_copy(self) -> Optional[np.ndarray]:
        """
        Returns the last successfully captured audio buffer as a float32 NumPy array.

        Attempts to get the current audio buffer first. If successful, updates the
        internal cache (`last_audio_copy`) and returns the new copy. If getting the
        current buffer fails (e.g., recorder not ready, empty buffer), it returns
        the previously cached buffer.

        Returns:
            A float32 NumPy array representing the audio, normalized to [-1.0, 1.0],
            or None if no audio has been successfully captured yet.
        """
        audio_copy = self.get_audio_copy()
        if audio_copy is not None and len(audio_copy) > 0:
            return audio_copy
        else:
            logger.debug('👂💾 Returning last known audio copy as current fetch failed or yielded empty.')
            return self.last_audio_copy

    def get_audio_copy(self) -> Optional[np.ndarray]:
        """
        Copies the current audio buffer from the recorder's frames.

        Retrieves the raw audio frames, concatenates them, converts to a float32
        NumPy array normalized to [-1.0, 1.0], and returns a deep copy. Updates
        `self.last_audio_copy` if successful. If the recorder is unavailable,
        frames are empty, or an error occurs, it returns the `last_audio_copy`.

        Returns:
            A deep copy of the current audio buffer as a float32 NumPy array,
            or the last known good copy if the current fetch fails, or None
            if no audio has ever been successfully captured.
        """
        if not self.recorder:
            logger.warning('👂⚠️ Cannot get audio copy: Recorder not initialized.')
            return self.last_audio_copy
        if not hasattr(self.recorder, 'frames'):
            logger.warning("👂⚠️ Cannot get audio copy: Recorder has no 'frames' attribute.")
            return self.last_audio_copy
        try:
            with self.recorder.frames_lock if hasattr(self.recorder, 'frames_lock') else threading.Lock():
                frames_data = list(self.recorder.frames)
            if not frames_data:
                logger.debug('👂💾 Recorder frames buffer is currently empty.')
                return self.last_audio_copy
            full_audio_array = np.frombuffer(b''.join(frames_data), dtype=np.int16)
            if full_audio_array.size == 0:
                logger.debug('👂💾 Recorder frames buffer resulted in empty array after join.')
                return self.last_audio_copy
            full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
            audio_copy = full_audio
            if audio_copy is not None and len(audio_copy) > 0:
                self.last_audio_copy = audio_copy
                logger.debug(f'👂💾 Successfully got audio copy (length: {len(audio_copy)} samples).')
            return audio_copy
        except Exception as e:
            logger.error(f'👂💥 Error getting audio copy: {e}', exc_info=True)
            return self.last_audio_copy

    def _create_recorder(self) -> None:
        """
        Internal helper to initialize the RealtimeSTT recorder instance
        (either local or client) with the specified configuration and callbacks.
        """

        def start_silence_detection():
            """Callback triggered when recorder detects start of silence (end of speech)."""
            self.set_silence(True)
            recorder_silence_start = self._get_recorder_param('speech_end_silence_start', None)
            self.silence_time = recorder_silence_start if recorder_silence_start else time.time()
            logger.debug(f'👂🤫 Silence detected (start_silence_detection called). Silence time set to: {self.silence_time}')

        def stop_silence_detection():
            """Callback triggered when recorder detects end of silence (start of speech)."""
            self.set_silence(False)
            self.silence_time = 0.0
            logger.debug('👂🗣️ Speech detected (stop_silence_detection called). Silence time reset.')

        def start_recording():
            """Callback triggered when recorder starts a new recording segment."""
            logger.info('👂▶️ Recording started.')
            self.set_silence(False)
            self.silence_time = 0.0
            if self.on_recording_start_callback:
                self.on_recording_start_callback()

        def stop_recording() -> bool:
            """
            Callback triggered when recorder stops a recording segment, just
            before final transcription might be generated.
            """
            logger.info('👂⏹️ Recording stopped.')
            audio_copy = self.get_last_audio_copy()
            if self.before_final_sentence:
                logger.debug('👂➡️ Calling before_final_sentence callback...')
                try:
                    result = self.before_final_sentence(audio_copy, self.realtime_text)
                    return result if isinstance(result, bool) else False
                except Exception as e:
                    logger.error(f'👂💥 Error in before_final_sentence callback: {e}', exc_info=True)
                    return False
            return False

        def on_partial(text: Optional[str]):
            """Callback triggered for real-time transcription updates."""
            if text is None:
                return
            self.realtime_text = text
            self.detect_potential_sentence_end(text)
            stripped_partial_user_text_new = strip_ending_punctuation(text)
            if stripped_partial_user_text_new != self.stripped_partial_user_text:
                self.stripped_partial_user_text = stripped_partial_user_text_new
                logger.info(f'👂📝 Partial transcription: {Colors.CYAN}{text}{Colors.RESET}')
                if self.realtime_transcription_callback:
                    self.realtime_transcription_callback(text)
                if USE_TURN_DETECTION and hasattr(self, 'turn_detection'):
                    self.turn_detection.calculate_waiting_time(text=text)
            else:
                logger.debug(f'👂📝 Partial transcription (no change after strip): {Colors.GRAY}{text}{Colors.RESET}')
        active_config = self.recorder_config.copy()
        active_config['on_realtime_transcription_update'] = on_partial
        active_config['on_turn_detection_start'] = start_silence_detection
        active_config['on_turn_detection_stop'] = stop_silence_detection
        active_config['on_recording_start'] = start_recording
        active_config['on_recording_stop'] = stop_recording

        def _pretty(v, max_len=60):
            if callable(v):
                return f'[callback: {v.__name__}]'
            if isinstance(v, str):
                one_line = v.replace('\n', ' ')
                return one_line[:max_len].rstrip() + ' [...]' if len(one_line) > max_len else one_line
            return v
        pretty_cfg = {k: _pretty(v) for k, v in active_config.items()}
        padded_cfg = textwrap.indent(json.dumps(pretty_cfg, indent=2), '    ')
        recorder_type = 'AudioToTextRecorderClient' if START_STT_SERVER else 'AudioToTextRecorder'
        logger.info(f'👂⚙️ Creating {recorder_type} with params:')
        print(Colors.apply(padded_cfg).blue)
        try:
            if START_STT_SERVER:
                self.recorder = AudioToTextRecorderClient(**active_config)
                self._set_recorder_param('use_wake_words', False)
            else:
                self.recorder = AudioToTextRecorder(**active_config)
                self._set_recorder_param('use_wake_words', False)
            logger.info(f'👂✅ {recorder_type} instance created successfully.')
        except Exception as e:
            logger.exception(f'👂🔥 Failed to create recorder: {e}')
            self.recorder = None

    def feed_audio(self, chunk: bytes, audio_meta_data: Optional[Dict[str, Any]]=None) -> None:
        """
        Feeds an audio chunk to the underlying recorder instance for processing.

        Args:
            chunk: A bytes object containing the raw audio data chunk.
            audio_meta_data: Optional dictionary containing metadata about the audio
                             (e.g., sample rate, channels), if required by the recorder.
        """
        if self.recorder and (not self.shutdown_performed):
            try:
                if START_STT_SERVER:
                    self.recorder.feed_audio(chunk)
                else:
                    self.recorder.feed_audio(chunk)
                logger.debug(f'👂🔊 Fed audio chunk of size {len(chunk)} bytes to recorder.')
            except Exception as e:
                logger.error(f'👂💥 Error feeding audio to recorder: {e}')
        elif not self.recorder:
            logger.warning('👂⚠️ Cannot feed audio: Recorder not initialized.')
        elif self.shutdown_performed:
            logger.debug('👂🚫 Cannot feed audio: Shutdown already performed.')

    def shutdown(self) -> None:
        """
        Shuts down the recorder instance, cleans up resources, and prevents
        further processing. Sets the `shutdown_performed` flag.
        """
        if not self.shutdown_performed:
            logger.info('👂🔌 Shutting down TranscriptionProcessor...')
            self.shutdown_performed = True
            if self.recorder:
                logger.info('👂🔌 Calling recorder shutdown()...')
                try:
                    self.recorder.shutdown()
                    logger.info('👂🔌 Recorder shutdown() method completed.')
                except Exception as e:
                    logger.error(f'👂💥 Error during recorder shutdown: {e}', exc_info=True)
                finally:
                    self.recorder = None
            else:
                logger.info('👂🔌 No active recorder instance to shut down.')
            if USE_TURN_DETECTION and hasattr(self, 'turn_detection') and hasattr(self.turn_detection, 'shutdown'):
                logger.info('👂🔌 Shutting down TurnDetection...')
                try:
                    self.turn_detection.shutdown()
                except Exception as e:
                    logger.error(f'👂💥 Error during TurnDetection shutdown: {e}', exc_info=True)
            logger.info('👂🔌 TranscriptionProcessor shutdown process finished.')
        else:
            logger.info('👂ℹ️ Shutdown already performed.')


# Cluster 4

def setup_logging(level: int=logging.INFO) -> None:
    """
    Configures the root logger for console output with a custom format and level.

    Sets up a `StreamHandler` for the root logger if no handlers are already present.
    Applies a `CustomTimeFormatter` to display timestamps as MM:SS.cs and uses
    ANSI colors for different log message parts (timestamp, logger name, level, message).
    This setup avoids modifying global logging state like the record factory or
    the global Formatter converter.

    Args:
        level: The minimum logging level for the root logger and the console handler
               (e.g., `logging.DEBUG`, `logging.INFO`). Defaults to `logging.INFO`.
    """
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        root_logger.setLevel(level)
        prefix = Colors.apply('🖥️').gray
        timestamp = Colors.apply('%(asctime)s').blue
        levelname = Colors.apply('%(levelname)-4.4s').green.bold
        message = Colors.apply('%(message)s')
        logger_name = Colors.apply('%(name)-10.10s').gray
        log_format = f'{timestamp} {logger_name} {levelname} {message}'
        formatter = CustomTimeFormatter(log_format)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        root_logger.addHandler(handler)

def format_timestamp_ns(timestamp_ns: int) -> str:
    """
    Formats a nanosecond timestamp into a human-readable HH:MM:SS.fff string.

    Args:
        timestamp_ns: The timestamp in nanoseconds since the epoch.

    Returns:
        A string formatted as hours:minutes:seconds.milliseconds.
    """
    seconds = timestamp_ns // 1000000000
    remainder_ns = timestamp_ns % 1000000000
    dt = datetime.fromtimestamp(seconds)
    time_str = dt.strftime('%H:%M:%S')
    milliseconds = remainder_ns // 1000000
    formatted_timestamp = f'{time_str}.{milliseconds:03d}'
    return formatted_timestamp

def parse_json_message(text: str) -> dict:
    """
    Safely parses a JSON string into a dictionary.

    Logs a warning if the JSON is invalid and returns an empty dictionary.

    Args:
        text: The JSON string to parse.

    Returns:
        A dictionary representing the parsed JSON, or an empty dictionary on error.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning('🖥️⚠️ Ignoring client message with invalid JSON')
        return {}

def log_status():
    nonlocal prev_status
    last_quick_answer_chunk_decayed = last_quick_answer_chunk and time.time() - last_quick_answer_chunk > TTS_FINAL_TIMEOUT and (time.time() - last_chunk_sent > TTS_FINAL_TIMEOUT)
    curr_status = (int(callbacks.tts_to_client), int(callbacks.tts_client_playing), int(callbacks.tts_chunk_sent), 1, int(callbacks.is_hot), int(callbacks.synthesis_started), int(app.state.SpeechPipelineManager.running_generation is not None), int(app.state.SpeechPipelineManager.is_valid_gen()), int(is_tts_finished), int(app.state.AudioInputProcessor.interrupted))
    if curr_status != prev_status:
        status = Colors.apply('🖥️🚦 State ').red
        logger.info(f'{status} ToClient {curr_status[0]}, ttsClientON {curr_status[1]}, ChunkSent {curr_status[2]}, hot {curr_status[4]}, synth {curr_status[5]} gen {curr_status[6]} valid {curr_status[7]} tts_q_fin {curr_status[8]} mic_inter {curr_status[9]}')
        prev_status = curr_status


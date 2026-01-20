# Cluster 2

def local_whisper_process(input_folder, crew_output_folder, transcript=None, subtitles=None, transcribe_flag=True):
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            input_video_path = os.path.join(input_folder, filename)
            logging.info(f'Processing video: {input_video_path}')
            if transcribe_flag:
                if transcript and subtitles:
                    initial_srt_path = os.path.join(crew_output_folder, f'{os.path.splitext(filename)[0]}_subtitles.srt')
                    with open(initial_srt_path, 'w') as srt_file:
                        srt_file.write(subtitles)
                else:
                    full_transcript, full_subtitles = transcribe_main(input_video_path)
                    initial_srt_path = os.path.join(crew_output_folder, f'{os.path.splitext(filename)[0]}_subtitles.srt')
                    with open(initial_srt_path, 'w') as srt_file:
                        srt_file.write(full_subtitles)
            else:
                initial_srt_path = os.path.join(crew_output_folder, f'{os.path.splitext(filename)[0]}.srt')
            if wait_for_file(initial_srt_path):
                whisper_output_dir = 'whisper_output'
                srt_files = [f for f in os.listdir(whisper_output_dir) if f.endswith('.srt')]
                txt_files = [f for f in os.listdir(whisper_output_dir) if f.endswith('.txt')]
                if srt_files and txt_files:
                    subtitles_file = os.path.join(whisper_output_dir, srt_files[0])
                    transcript_file = os.path.join(whisper_output_dir, txt_files[0])
                    with open(transcript_file, 'r') as file:
                        transcript = file.read()
                    with open(subtitles_file, 'r') as file:
                        subtitles = file.read()
                    for srt_filename in sorted(os.listdir(crew_output_folder)):
                        if srt_filename.startswith('new_file_return_subtitles') and srt_filename.endswith('.srt'):
                            subtitle_file_path = os.path.join(crew_output_folder, srt_filename)
                else:
                    logging.error('No .srt or .txt files found in the whisper_output directory.')
            else:
                logging.error(f'Failed to verify the readiness of subtitles file: {initial_srt_path}')
    logging.info(f'local_transcribe.py completed')

def transcribe_main(file):
    plain = True
    srt = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model('medium.en').to(DEVICE)
    result, transcript, subtitles = transcribe_file(model, srt, plain, file)
    return (transcript, subtitles)

def transcribe_file(model, srt, plain, file):
    input_file_path = Path(file)
    logging.info(f'Transcribing file: {input_file_path}\n')
    output_dir = Path('whisper_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    result = model.transcribe(str(input_file_path), fp16=False, verbose=False, language='en')
    output_file_name = input_file_path.stem
    if plain:
        txt_path = output_dir / f'{output_file_name}.txt'
        logging.info(f'Creating text file: {txt_path}')
        with open(txt_path, 'w', encoding='utf-8') as txt:
            txt.write(result['text'])
        transcript = result['text']
    if srt:
        logging.info(f'Creating SRT file')
        srt_writer = get_writer('srt', str(output_dir))
        srt_writer(result, output_file_name)
        srt_path = output_dir / f'{output_file_name}.srt'
        with open(srt_path, 'r', encoding='utf-8') as srt_file:
            subtitles = srt_file.read()
    return (result, transcript, subtitles)

def wait_for_file(filepath):
    """
    This function waits for a file to be available before proceeding.

    Args:
        filepath: Path to the file to wait for
    """
    lock = lockfile.FileLock(filepath)
    while not lock.i_am_locking():
        try:
            lock.acquire(timeout=1)
        except lockfile.LockTimeout:
            pass
    return True


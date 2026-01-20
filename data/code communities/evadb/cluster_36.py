# Cluster 36

def try_to_import_pytube():
    try:
        import pytube
    except ImportError:
        raise ValueError('Could not import pytube python package.\n                Please install it with `pip install -r requirements.txt`.')

def generate_summary(cursor: evadb.EvaDBCursor):
    """Generate summary of a video transcript if it is too long (exceeds llm token limits)

    Args:
        cursor (EVADBCursor): evadb api cursor.
    """
    transcript_list = cursor.query('SELECT text FROM Transcript;').df()['transcript.text']
    if len(transcript_list) == 1:
        summary = transcript_list[0]
        df = pd.DataFrame([{'summary': summary}])
        df.to_csv(SUMMARY_PATH)
        cursor.query('DROP TABLE IF EXISTS Summary;').execute()
        cursor.query('CREATE TABLE IF NOT EXISTS Summary (summary TEXT(100));').execute()
        cursor.query(f"LOAD CSV '{SUMMARY_PATH}' INTO Summary;").execute()
        return
    generate_summary_text_query = "SELECT ChatGPT('summarize the video in detail', text) FROM Transcript;"
    responses = cursor.query(generate_summary_text_query).df()['chatgpt.response']
    summary = ''
    for r in responses:
        summary += f'{r} \n'
    df = pd.DataFrame([{'summary': summary}])
    df.to_csv(SUMMARY_PATH)
    need_to_summarize = len(summary) > MAX_CHUNK_SIZE
    while need_to_summarize:
        partitioned_summary = partition_summary(summary)
        df = pd.DataFrame([{'summary': partitioned_summary}])
        df.to_csv(SUMMARY_PATH)
        cursor.query('DROP TABLE IF EXISTS Summary;').execute()
        cursor.query('CREATE TABLE IF NOT EXISTS Summary (summary TEXT(100));').execute()
        cursor.query(f"LOAD CSV '{SUMMARY_PATH}' INTO Summary;").execute()
        generate_summary_text_query = "SELECT ChatGPT('summarize in detail', summary) FROM Summary;"
        responses = cursor.query(generate_summary_text_query).df()['chatgpt.response']
        summary = ' '.join(responses)
        if len(summary) <= MAX_CHUNK_SIZE:
            need_to_summarize = False
    cursor.query('DROP TABLE IF EXISTS Summary;').execute()
    cursor.query('CREATE TABLE IF NOT EXISTS Summary (summary TEXT(100));').execute()
    cursor.query(f"LOAD CSV '{SUMMARY_PATH}' INTO Summary;").execute()

def partition_summary(prev_summary: str):
    """Summarize a summary if a summary is too large.

    Args:
        prev_summary (str): previous summary that is too large.

    Returns:
        List: a list of partitioned summary
    """
    k = 2
    while True:
        if len(prev_summary) / k <= MAX_CHUNK_SIZE:
            break
        else:
            k += 1
    chunk_size = int(len(prev_summary) / k)
    new_summary = [{'summary': prev_summary[i:i + chunk_size]} for i in range(0, len(prev_summary), chunk_size)]
    if len(new_summary[-1]['summary']) < 30:
        new_summary.pop()
    return new_summary

def generate_response(cursor: evadb.EvaDBCursor, question: str) -> str:
    """Generates question response with llm.

    Args:
        cursor (EVADBCursor): evadb api cursor.
        question (str): question to ask to llm.

    Returns
        str: response from llm.
    """
    if len(cursor.query('SELECT text FROM Transcript;').df()['transcript.text']) == 1:
        return cursor.query(f"SELECT ChatGPT('{question}', text) FROM Transcript;").df()['chatgpt.response'][0]
    else:
        if not os.path.exists(SUMMARY_PATH):
            generate_summary(cursor)
        return cursor.query(f"SELECT ChatGPT('{question}', summary) FROM Summary;").df()['chatgpt.response'][0]

def generate_blog_post(cursor: evadb.EvaDBCursor):
    """Generates blog post.

    Args:
        cursor (EVADBCursor): evadb api cursor.
    """
    to_generate = str(input('\nWould you like to generate a blog post based on the video? (yes/no): '))
    if to_generate.lower() == 'yes' or to_generate.lower() == 'y':
        print('⏳ Generating blog post (may take a while)...')
        if not os.path.exists(SUMMARY_PATH):
            generate_summary(cursor)
        sections = generate_blog_sections(cursor)
        title_query = 'generate a creative title of a blog post from the transcript'
        generate_title_rel = cursor.query(f"SELECT ChatGPT('{title_query}', summary) FROM Summary;")
        blog = '# ' + generate_title_rel.df()['chatgpt.response'][0].replace('"', '')
        i = 1
        for section in sections:
            print(f'--⏳ Generating body ({i}/{len(sections)}) titled {section}...')
            if 'introduction' in section.lower():
                section_query = f'write a section about {section} from transcript'
                section_prompt = 'generate response in markdown format and highlight important technical terms with hyperlinks'
            elif 'conclusion' in section.lower():
                section_query = 'write a creative conclusion from transcript'
                section_prompt = 'generate response in markdown format'
            else:
                section_query = f'write a single detailed section about {section} from transcript'
                section_prompt = 'generate response in markdown format with information from the internet'
            generate_section_rel = cursor.query(f"SELECT ChatGPT('{section_query}', summary, '{section_prompt}') FROM Summary;")
            generated_section = generate_section_rel.df()['chatgpt.response'][0]
            print(generated_section)
            blog += '\n' + generated_section + '\n'
            i += 1
        source_query = 'generate a short list of keywords for the transcript with hyperlinks'
        source_prompt = 'generate response in markdown format'
        print('--⏳ Wrapping up...')
        generate_source_rel = cursor.query(f"SELECT ChatGPT('{source_query}', summary, '{source_prompt}') FROM Summary;")
        blog += '\n## Sources\n' + generate_source_rel.df()['chatgpt.response'][0]
        print(blog)
        if os.path.exists(BLOG_PATH):
            os.remove(BLOG_PATH)
        with open(BLOG_PATH, 'w') as file:
            file.write(blog)
        print(f'✅ blog post is saved to file {os.path.abspath(BLOG_PATH)}')

def generate_blog_sections(cursor: evadb.EvaDBCursor) -> List:
    """Generates logical sections of the blog post.

    Args:
        cursor (EVADBCursor): evadb api cursor.

    Returns
        List: list of blog sections
    """
    sections_query = 'list 7 logical sections of a blog post from the transcript as a python list'
    sections_string = str(cursor.query(f"SELECT ChatGPT('{sections_query}', summary) FROM Summary;").df()['chatgpt.response'][0])
    begin = sections_string.find('[')
    end = sections_string.find(']')
    assert begin != -1 and end != -1, 'cannot infer blog sections.'
    sections_string = sections_string[begin + 1:end]
    sections_string = sections_string.replace('\n', '')
    sections_string = sections_string.replace('\t', '')
    sections_string = sections_string.replace('"', '')
    sections = sections_string.split(',')
    for i in range(len(sections)):
        sections[i] = sections[i].strip()
    print(sections)
    return sections

def receive_user_input() -> Dict:
    """Receives user input.

    Returns:
        user_input (dict): global configurations
    """
    print('🔮 Welcome to EvaDB! This app lets you ask questions on any local or YouTube online video.\nYou will only need to supply a Youtube URL and an OpenAI API key.\n')
    from_youtube = str(input("📹 Are you querying an online Youtube video or a local video? ('yes' for online/ 'no' for local): ")).lower() in ['y', 'yes']
    user_input = {'from_youtube': from_youtube}
    if from_youtube:
        video_link = str(input('🌐 Enter the URL of the YouTube video (press Enter to use our default Youtube video URL): '))
        if video_link == '':
            video_link = DEFAULT_VIDEO_LINK
        user_input['video_link'] = video_link
    else:
        video_local_path = str(input('💽 Enter the local path to your video (press Enter to use our demo video): '))
        if video_local_path == '':
            video_local_path = DEFAULT_VIDEO_PATH
        user_input['video_local_path'] = video_local_path
    try:
        api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        api_key = str(input('🔑 Enter your OpenAI key: '))
        os.environ['OPENAI_API_KEY'] = api_key
    return user_input

def download_youtube_video_transcript(video_link: str):
    """Downloads a YouTube video's transcript.

    Args:
        video_link (str): url of the target YouTube video.
    """
    video_id = extract.video_id(video_link)
    print('⏳ Transcript download in progress...')
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    print('✅ Video transcript downloaded successfully.')
    return transcript

def group_transcript(transcript: dict):
    """Group video transcript elements when they are too short.

    Args:
        transcript (dict): downloaded video transcript as a dictionary.

    Returns:
        str: full transcript as a single string.
    """
    new_line = ''
    for line in transcript:
        new_line += ' ' + line['text']
    return new_line

def download_youtube_video_from_link(video_link: str):
    """Downloads a YouTube video from url.

    Args:
        video_link (str): url of the target YouTube video.
    """
    yt = YouTube(video_link).streams.filter(file_extension='mp4', progressive='True').first()
    try:
        print('⏳ video download in progress...')
        yt.download(filename=ONLINE_VIDEO_PATH)
    except Exception as e:
        print(f'⛔️ Video download failed with error: \n{e}')
    print('✅ Video downloaded successfully.')

def generate_online_video_transcript(cursor: evadb.EvaDBCursor) -> str:
    """Extracts speech from video for llm processing.

    Args:
        cursor (EVADBCursor): evadb api cursor.

    Returns:
        str: video transcript text.
    """
    print('\n⏳ Analyzing YouTube video. This may take a while...')
    cursor.query('DROP TABLE IF EXISTS youtube_video;').execute()
    cursor.query(f"LOAD VIDEO '{ONLINE_VIDEO_PATH}' INTO youtube_video;").execute()
    cursor.query('DROP TABLE IF EXISTS youtube_video_text;').execute()
    cursor.query('CREATE TABLE IF NOT EXISTS youtube_video_text AS SELECT SpeechRecognizer(audio) FROM youtube_video;').execute()
    print('✅ Video analysis completed.')
    raw_transcript_string = cursor.query('SELECT text FROM youtube_video_text;').df()['youtube_video_text.text'][0]
    return raw_transcript_string

def generate_local_video_transcript(cursor: evadb.EvaDBCursor, video_path: str) -> str:
    """Extracts speech from video for llm processing.

    Args:
        cursor (EVADBCursor): evadb api cursor.
        video_path (str): video path.

    Returns:
        str: video transcript text.
    """
    print(f'\n⏳ Analyzing local video from {video_path}. This may take a while...')
    cursor.query('DROP TABLE IF EXISTS local_video;').execute()
    cursor.query(f"LOAD VIDEO '{video_path}' INTO local_video;").execute()
    cursor.query('DROP TABLE IF EXISTS local_video_text;').execute()
    cursor.query('CREATE TABLE IF NOT EXISTS local_video_text AS SELECT SpeechRecognizer(audio) FROM local_video;').execute()
    print('✅ Video analysis completed.')
    raw_transcript_string = cursor.query('SELECT text FROM local_video_text;').df()['local_video_text.text'][0]
    return raw_transcript_string

def partition_transcript(raw_transcript: str):
    """Group video transcript elements when they are too large.

    Args:
        transcript (str): downloaded video transcript as a raw string.

    Returns:
        List: a list of partitioned transcript
    """
    if len(raw_transcript) <= MAX_CHUNK_SIZE:
        return [{'text': raw_transcript}]
    k = 2
    while True:
        if len(raw_transcript) / k <= MAX_CHUNK_SIZE:
            break
        else:
            k += 1
    chunk_size = int(len(raw_transcript) / k)
    partitioned_transcript = [{'text': raw_transcript[i:i + chunk_size]} for i in range(0, len(raw_transcript), chunk_size)]
    if len(partitioned_transcript[-1]['text']) < 30:
        partitioned_transcript.pop()
    return partitioned_transcript

def cleanup():
    """Removes any temporary file / directory created by EvaDB."""
    if os.path.exists('evadb_data'):
        shutil.rmtree('evadb_data')

def generate_script(cursor: evadb.EvaDBCursor, df: pd.DataFrame, question: str) -> str:
    """Generates script with llm.

    Args:
        cursor (EVADBCursor): evadb api cursor.
        question (str): question to ask to llm.

    Returns
        str: script generated by llm.
    """
    all_columns = list(df)
    df[all_columns] = df[all_columns].astype(str)
    prompt = f'There is a dataframe in pandas (python). The name of the\n            dataframe is df. This is the result of print(df.head()):\n            {str(df.head())}. Return a python script with comments to get the answer to the following question: {question}. Do not write code to load the CSV file.'
    question_df = pd.DataFrame([{'prompt': prompt}])
    question_df.to_csv(QUESTION_PATH)
    cursor.drop_table('Question', if_exists=True).execute()
    cursor.query('CREATE TABLE IF NOT EXISTS Question (prompt TEXT(50));').execute()
    cursor.load(QUESTION_PATH, 'Question', 'csv').execute()
    pd.set_option('display.max_colwidth', None)
    query = cursor.table('Question').select('ChatGPT(prompt)')
    script_body = query.df()['chatgpt.response'][0]
    return script_body

def run_script(script_body: str, user_input: Dict):
    """Runs script generated by llm.

    Args:
        script_body (str): script generated by llm.
        user_input (Dict): user input.
    """
    absolute_csv_path = os.path.abspath(user_input['csv_path'])
    absolute_script_path = os.path.abspath(SCRIPT_PATH)
    print(absolute_csv_path)
    load_df = f"import pandas as pd\ndf = pd.read_csv('{absolute_csv_path}')\n"
    script_body = load_df + script_body
    with open(absolute_script_path, 'w+') as script_file:
        script_file.write(script_body)
    subprocess.run(['python', absolute_script_path])

